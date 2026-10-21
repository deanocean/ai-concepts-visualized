/* LSTM のゲート（シラバス §22；CEC は §27 キーワード）
   一個純量記憶單元 c，記住「上一個提到的人是女性(+)還是男性(−)」，到空格時填 she / he。
   三個閘門都是 sigmoid（0〜1），逐詞計算；可把任一閘門固定成 1，看少了它會出什麼事。
   權重是為了示範而設計的小數字（真實模型的閘門也會看上一步的 h，數字全由訓練學到）。
   文字在 locales/zh/lstm.js、locales/ja/lstm.js（例句是英文，兩種語言共用，留在程式裡）。 */
Atlas.define("lstm", {
  mount(bench, ctx) {
    const { h, s, fmt, fmtP, ui } = Atlas;
    const { t } = ctx;
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
    const GATE = { f: t("gateF"), i: t("gateI"), o: t("gateO") };
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
    const sentSeg = ui.seg({ label: t("sentLabel"), value: sent,
      options: [{ value: "A", label: t("sentA") }, { value: "B", label: t("sentB") }],
      onChange: (v) => { sent = v; player.pause(); pos = 0; render(); } });
    const modeSeg = ui.seg({ label: t("modeLabel"), value: mode,
      options: [{ value: "n", label: t("modeN") }, { value: "f", label: t("modeF") }, { value: "i", label: t("modeI") }, { value: "o", label: t("modeO") }],
      onChange: (v) => { mode = v; render(); } });
    const T = 8;
    const player = ui.player(ctx, () => { if (pos >= T) return false; pos++; render(); return pos < T; }, 900);
    const prevBtn = ui.btn(t("btnPrev"), () => { player.pause(); if (pos > 0) { pos--; render(); } });
    const nextBtn = ui.btn(t("btnNext"), () => { player.pause(); if (pos < T) { pos++; render(); } }, { primary: true });
    const resetBtn = ui.btn(t("btnReset"), () => { player.pause(); pos = 0; render(); });

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
      h("div", { class: "field" }, h("span", { class: "field__label" }, t("lineLabel")), lineWrap),
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
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("cellAria"), style: `display:block;width:100%;min-width:${W * 0.78}px;max-width:${W}px` });
      const j = pos - 1, st = j >= 0 ? R[j] : null;
      const on = !!st, ink = on ? "var(--ink)" : "var(--faint)", wire = on ? "var(--muted)" : "var(--line-strong)";
      const yC = 52, xF = 220, xA = 400, yMul = 112, yG = 172, xI = 335, xGc = 465, xO = 590, xOut = 680;

      // memory highway
      svg.append(s("line", { x1: 118, y1: yC, x2: 548, y2: yC, stroke: "var(--line-strong)", "stroke-width": 7, "stroke-linecap": "round", "stroke-opacity": 0.55 }));
      svg.append(s("text", { x: (xF + xA) / 2, y: 18, "text-anchor": "middle", "font-size": 11.5, class: "t-muted", text: t("highway") }));
      const box = (x, y, w, hh, val, label, sub, max = 1.2) => {
        svg.append(s("rect", { x, y, width: w, height: hh, rx: 8, fill: on && val != null ? Atlas.divColor(val, max) : "var(--panel-2)", stroke: "var(--line-strong)" }));
        svg.append(s("text", { x: x + w / 2, y: y - 6, "text-anchor": "middle", "font-size": 12, "font-weight": 700, text: label }));
        svg.append(s("text", { x: x + w / 2, y: y + hh / 2 + 5, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: on && val != null && Math.abs(val) > 0.75 * max ? "var(--panel)" : ink, text: on && val != null ? fmt(val, 2) : "?" }));
        if (sub) svg.append(s("text", { x: x + w / 2, y: y + hh + 15, "text-anchor": "middle", "font-size": 11, class: "t-muted", text: sub }));
      };
      box(18, yC - 18, 96, 36, st ? st.cPrev : null, t("oldC") + " c", t("oldCSub"));
      box(552, yC - 18, 96, 36, st ? st.c : null, t("newC") + " c", "");
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
        svg.append(s("text", { x, y: y + 48, "text-anchor": "middle", "font-size": 12.5, class: "t-mono", fill: ink, text: on ? (isFix ? t("fixed1") : fmt(val, 3)) : "?" }));
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
      svg.append(s("text", { x: xGc, y: yG + 17, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: on && Math.abs(st.g) > 0.75 ? "var(--panel)" : "var(--ink)", text: t("gateG") + " g" }));
      svg.append(s("text", { x: xGc, y: yG + 42, "text-anchor": "middle", "font-size": 14, class: "t-mono", fill: on && Math.abs(st.g) > 0.75 ? "var(--panel)" : ink, text: on ? fmt(st.g, 3) : "?" }));
      gate(xO, "o", st ? st.o : 0);
      // h output
      const hv = st ? st.h : null;
      svg.append(s("rect", { x: xO + 26, y: 262, width: 88, height: 32, rx: 8, fill: on ? Atlas.divColor(hv, 1) : "var(--panel-2)", stroke: "var(--line-strong)" }));
      svg.append(s("text", { x: xO + 70, y: 283, "text-anchor": "middle", "font-size": 13.5, class: "t-mono", fill: on && Math.abs(hv) > 0.75 ? "var(--panel)" : ink, text: on ? fmt(hv, 2) : "?" }));
      svg.append(s("text", { x: xO + 120, y: 283, "font-size": 12, "font-weight": 700, text: t("outK") + " h" }));
      // input bus
      const yBus = 252;
      svg.append(s("line", { x1: 118, y1: yBus, x2: xO, y2: yBus, stroke: wire, "stroke-width": 1.4 }));
      [xF, xI, xGc, xO].forEach((x) => arrow(svg, x, yBus, x, yG + 57, wire, 1.4));
      svg.append(s("rect", { x: 10, y: yBus - 20, width: 104, height: 40, rx: 8, fill: "var(--hot-soft)", stroke: "var(--hot)" }));
      svg.append(s("text", { x: 62, y: yBus - 26, "text-anchor": "middle", "font-size": 12, "font-weight": 700, text: t("wordNow") }));
      svg.append(s("text", { x: 62, y: yBus + 6, "text-anchor": "middle", "font-size": 15, class: "t-mono", "font-weight": 600, text: st ? st.w : "—" }));
      // tags of the word
      if (st) {
        const tag = st.p ? t(st.sx > 0 ? "tagNameF" : "tagNameM") : st.q ? t("tagBlank") : t("tagOther");
        svg.append(s("text", { x: 62, y: yBus + 36, "text-anchor": "middle", "font-size": 11.5, fill: "var(--hot)", text: tag }));
      }
      cellWrap.replaceChildren(svg);
    }

    /* ---------- the whole sentence ---------- */
    function drawLine(R) {
      const COLW = 78, LEFT = 82, W = LEFT + COLW * T + 8;
      const rows = [[t("rowWord"), 18], [GATE.f + " f", 50], [GATE.i + " i", 78], [GATE.o + " o", 106], [t("rowC") + " c", 142], [t("outK") + " h", 182]];
      const H = 206;
      const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("lineAria"), style: `display:block;width:100%;min-width:${W * 0.8}px;max-width:${W}px` });
      rows.forEach(([label, y], r) => svg.append(s("text", { x: 4, y: y + 4, "font-size": 11.5, "font-weight": 700, class: r === 0 ? "t-muted" : "", text: label })));
      R.forEach((st, j) => {
        const x = LEFT + COLW * j, cx = x + COLW / 2, on = j < pos, cur = j === pos - 1;
        const col = s("g", { style: "cursor:pointer" });
        col.append(s("rect", { x: x + 2, y: 2, width: COLW - 4, height: H - 4, rx: 8, fill: cur ? "var(--hot-soft)" : "transparent", stroke: cur ? "var(--hot)" : "none" }));
        col.append(s("text", { x: cx, y: 22, "text-anchor": "middle", "font-size": 13, class: "t-mono", "font-weight": cur ? 700 : 400, fill: on ? "var(--ink)" : "var(--faint)", text: st.w }));
        ["f", "i", "o"].forEach((k, n) => {
          const y = rows[n + 1][1];
          col.append(s("rect", { x: cx - 30, y: y - 7, width: 60, height: 10, rx: 4, fill: "var(--panel-2)", stroke: "var(--line)" }));
          if (on) col.append(s("rect", { x: cx - 30, y: y - 7, width: 60 * st[k], height: 10, rx: 4, fill: fixed(k) ? "var(--hot)" : "var(--accent)" }));
          if (on) col.append(s("text", { x: cx, y: y + 16, "text-anchor": "middle", "font-size": 9.5, class: "t-mono t-muted", text: fmt(st[k], 2) }));
        });
        const cell = (y, v) => {
          col.append(s("rect", { x: cx - 30, y: y - 14, width: 60, height: 26, rx: 6, fill: on ? Atlas.divColor(v, 1.2) : "var(--panel-2)", stroke: "var(--line-strong)" }));
          col.append(s("text", { x: cx, y: y + 4, "text-anchor": "middle", "font-size": 12, class: "t-mono", fill: on && Math.abs(v) > 0.9 ? "var(--panel)" : on ? "var(--ink)" : "var(--faint)", text: on ? fmt(v, 2) : "?" }));
        };
        cell(rows[4][1], st.c); cell(rows[5][1], st.h);
        col.addEventListener("click", () => { player.pause(); pos = j + 1; render(); });
        svg.append(col);
      });
      lineWrap.replaceChildren(svg);
    }

    /* ---------- status & calc ---------- */
    function render() {
      const R = run(sent, mode), Rn = run(sent, "n");
      const S = SENT[sent];
      sentLine.innerHTML = t("sentLine", { s: `<b class="mono">${S.label}</b>`, a: `<b class="mono">${S.answer}</b>` });
      drawCell(R); drawLine(R);
      const set = (k, msg) => { status.className = "status status--" + k; status.textContent = msg; };
      const j = pos - 1, st = R[j];
      const row = (k, b) => `<span class="calc-row"><span class="k">${k}</span> ${b}</span>`;
      if (pos === 0) {
        set("info", t("statusStart"));
        calc.innerHTML = `<span class="k">${t("calcStart")}</span>`;
      } else {
        const fz = (k, v) => (fixed(k) ? `${v} → <span class="hl">${t("fixedTo1")}</span>` : v);
        calc.innerHTML =
          row(t("calcFeatK"), t("calcFeat", { p: st.p, sx: fmt(st.sx, 0), q: st.q, r: fmt(st.r, 1) })) +
          row(GATE.f, t("calcF", { z: fmt(st.zf, 0), v: fz("f", fmt(sig(st.zf), 3)) })) +
          row(GATE.i, t("calcI", { z: fmt(st.zi, 0), v: fz("i", fmt(sig(st.zi), 3)) })) +
          row(t("gateG"), t("calcG", { z: fmt(st.zg, 2), v: fmt(st.g, 3) })) +
          row(GATE.o, t("calcO", { z: fmt(st.zo, 0), v: fz("o", fmt(sig(st.zo), 3)) })) +
          row(t("newC"), t("calcC", { f: fmt(st.f, 3), cp: fmtP(st.cPrev, 3), i: fmt(st.i, 3), g: fmtP(st.g, 3), c: fmt(st.c, 3) })) +
          row(t("outK"), `h = o × tanh(c) = ${fmt(st.o, 3)} × ${fmt(Math.tanh(st.c), 3)} = <span class="hl">${fmt(st.h, 3)}</span>`) +
          (st.q ? row(t("predK"), t("calcPred", { z: fmt(6 * st.h, 2), p: fmt(pShe(st.h), 3), q: fmt(1 - pShe(st.h), 3) })) : "");

        if (st.q) {
          const P = pShe(st.h), pred = P >= 0.5 ? "she" : "he", conf = Math.max(P, 1 - P), pct = Math.round(conf * 100);
          const ok = pred === S.answer;
          const cN = Rn[j].c;
          if (mode === "o") set("warn", t("statusOutOpen", { pred, pct, h: fmt(R[1].h, 2) }));
          else if (mode === "f" && conf < 0.65) set("bad", t("statusFMixed", { p: fmt(P, 2), c: fmt(st.c, 2) }));
          else if (mode === "i" && !ok) set("bad", t("statusIWrong", { pred, pct, c0: fmt(R[0].c, 2), c: fmt(st.c, 2) }));
          else if (mode === "i") set("warn", t("statusILucky", { pred, c: fmt(st.c, 2), cn: fmt(cN, 2) }));
          else set(ok ? "ok" : "bad", t(ok ? "statusPredOk" : "statusPredBad", { o: fmt(st.o, 2), pred, pct }));
        } else if (st.p) {
          const who = t(st.sx > 0 ? "whoF" : "whoM");
          if (mode === "f" && j > 0) set("warn", t("statusNameFixF", { w: st.w, who, c: fmt(st.c, 2) }));
          else set("info", t(mode === "f" ? "statusNameF0" : "statusName", { w: st.w, f: fmt(st.f, 2), i: fmt(st.i, 2), who, c: fmt(st.c, 2) }));
        } else {
          if (mode === "i") set("warn", t("statusOtherI", { w: st.w, g: fmt(st.g, 2), cp: fmt(st.cPrev, 2), c: fmt(st.c, 2) }));
          else if (mode === "o") set("warn", t("statusOtherO", { w: st.w, h: fmt(st.h, 2) }));
          else set("info", t("statusOther", { w: st.w, f: fmt(st.f, 3), i: fmt(st.i, 3), cp: fmt(st.cPrev, 2), c: fmt(st.c, 2), h: fmt(st.h, 2) }));
        }
      }
      note.innerHTML = t("note");
      prevBtn.disabled = pos === 0;
      nextBtn.disabled = pos >= T;
    }
    render();
    ctx.onTheme(render);
    return { destroy() { player.pause(); } };
  },
});
