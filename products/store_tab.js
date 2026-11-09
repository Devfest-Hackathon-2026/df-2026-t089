// ⑤ 店モデル: 歩いて店を再現できるかのシミュレーション（dashboard.html のタブ。計算は store_sim.py）
const STORE = { cfg:{ seed:1, mode:"scan", recog:"real", drift:1, depth:1, px50:30, speed:0.25, use_themes:"1", themed:"1" }, res:null, batch:null, i:0, draw:false, route:[], yaw:-0.6, pitch:0.95, zoom:1, timer:null };
const KIND = { island:["島ゴンドラ",0], wall:["壁面ゴンドラ",1], case:["冷蔵・冷凍ケース",2], end:["エンド什器",3], register:["レジ",4] };
const PAL = { light:["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"], dark:["#3987e5","#d95926","#199e70","#c98500","#d55181"] };
const scheme = () => matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
const kindColor = (k) => PAL[scheme()][KIND[k][1]];
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (v) => v == null ? "-" : Math.round(v * 100) + "%";

function store(m) {
  const c = STORE.cfg;
  const sel = (k, label, opts) => `<label>${label} <select data-k="${k}">${opts.map(([v, t]) => `<option value="${v}" ${String(c[k]) == String(v) ? "selected" : ""}>${t}</option>`).join("")}</select></label>`;
  const num = (k, label, min, max, step) => `<label>${label} <input data-k="${k}" type="number" min="${min}" max="${max}" step="${step}" value="${c[k] ?? ""}" style="width:68px"></label>`;
  m.append($(`<div class="card row">
    ${num("seed", "店の番号", 1, 9999, 1)}
    ${sel("mode", "歩き方", [["scan", "棚を見回す"], ["forward", "前を向いたまま"]])}
    ${sel("recog", "認識", [["real", "実測の誤り"], ["perfect", "完璧"]])}
    ${num("speed", "速さ m/s", 0.1, 1.5, 0.05)}
    ${num("drift", "位置のずれ 倍", 0, 10, 0.5)}
    ${num("depth", "深度の誤差 倍", 0, 5, 0.5)}
    ${num("px50", "検出できる大きさ px", 10, 120, 5)}
    ${num("acc", "分類の正解率", 0, 1, 0.05)}
    ${sel("use_themes", "テーマ制約", [["1", "使う"], ["0", "使わない"]])}
    ${sel("themed", "店のテーマ", [["1", "あり"], ["0", "なし"]])}
    <button id="sim-run">実行</button>
    <button id="sim-draw">ルートを描く</button>
    <span class="mute" id="sim-note"></span></div>`));
  m.append($(`<div class="card row"><button id="sim-play">▶</button><input id="sim-t" type="range" min="0" value="0" style="flex:1"><span id="sim-clock" class="num"></span></div>`));
  m.append($(`<div class="grid" id="sim-tiles" style="margin-bottom:12px"></div>`));
  m.append($(`<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px;margin-bottom:12px">
    <div class="card"><b>3D（色つき = 推定、点線 = 正解）</b><span class="mute"> ドラッグで回転、ホイールで拡大</span><canvas id="sim-3d" width="760" height="520" style="width:100%;cursor:grab"></canvas></div>
    <div class="card"><b>上から見た図</b><span class="mute" id="sim-hint"> 什器に重ねると推定した商品が出ます</span><canvas id="sim-plan" width="760" height="520" style="width:100%"></canvas><div id="sim-legend" class="row" style="gap:12px;font-size:12px"></div></div></div>`));
  m.append($(`<div class="card"><b>歩くほど再現できるか</b><div id="sim-chart"></div></div>`));
  m.append($(`<div class="card"><b>店と条件を変えた平均</b> <span class="mute" id="sim-bnote"></span><div id="sim-batch"></div></div>`));
  m.append($(`<div class="card"><b>記述言語で書いた推定結果</b><pre class="log" id="sim-text"></pre></div>`));
  m.append($(`<div class="card"><b>棚ごとの主な商品（正解と推定）</b><div id="sim-rows" style="max-height:360px;overflow:auto"></div></div>`));
  m.querySelectorAll("[data-k]").forEach(e => e.onchange = () => { if (e.value === "") delete c[e.dataset.k]; else c[e.dataset.k] = e.type == "number" ? +e.value : e.value; });
  document.getElementById("sim-run").onclick = () => simRun(STORE.route);
  document.getElementById("sim-draw").onclick = (e) => {
    STORE.draw = !STORE.draw;
    e.target.textContent = STORE.draw ? "このルートで実行" : "ルートを描く";
    document.getElementById("sim-hint").textContent = STORE.draw ? " クリックした順に歩きます（店の中をクリック）" : " 什器に重ねると推定した商品が出ます";
    if (STORE.draw) { STORE.route = []; simDraw(); } else if (STORE.route.length > 1) simRun(STORE.route);
  };
  document.getElementById("sim-t").oninput = (e) => { STORE.i = +e.target.value; simDraw(); };
  document.getElementById("sim-play").onclick = simPlay;
  simDrag(document.getElementById("sim-3d"));
  simPlan(document.getElementById("sim-plan"));
  fetch("/api/store_sim_batch").then(r => r.json()).then(b => { STORE.batch = b; simBatch(); });
  if (STORE.res) simShow(); else simRun([]);
}

async function simRun(route) {
  const note = document.getElementById("sim-note");
  note.textContent = "計算中…";
  const q = new URLSearchParams(STORE.cfg);
  if (route.length > 1) q.set("route", route.map(p => p.map(v => v.toFixed(2)).join(",")).join(";"));
  const r = await fetch("/api/store_sim?" + q);
  if (!r.ok) { note.textContent = await r.text(); return; }
  STORE.res = await r.json();
  STORE.i = STORE.res.frames.length - 1;
  note.textContent = "";
  simShow();
}

function simPlay() {
  const b = document.getElementById("sim-play");
  if (STORE.timer) { clearInterval(STORE.timer); STORE.timer = null; b.textContent = "▶"; return; }
  if (STORE.i >= STORE.res.frames.length - 1) STORE.i = 0;
  b.textContent = "⏸";
  STORE.timer = setInterval(() => {
    if (!document.getElementById("sim-t") || STORE.i >= STORE.res.frames.length - 1) { clearInterval(STORE.timer); STORE.timer = null; if (document.getElementById("sim-play")) document.getElementById("sim-play").textContent = "▶"; return; }
    STORE.i++; document.getElementById("sim-t").value = STORE.i; simDraw();
  }, 120);
}

const simSnap = () => { const r = STORE.res; let s = r.snaps[0]; for (const x of r.snaps) if (x.t <= STORE.i) s = x; return s; };

function simShow() {
  const r = STORE.res, t = document.getElementById("sim-t");
  t.max = r.frames.length - 1; t.value = STORE.i;
  document.getElementById("sim-text").textContent = r.text;
  document.getElementById("sim-legend").innerHTML = Object.entries(KIND).map(([k, v]) => `<span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${kindColor(k)};margin-right:4px"></i>${v[0]}</span>`).join("");
  document.getElementById("sim-rows").innerHTML = `<table><tr><th>什器</th><th>正解</th><th>推定</th><th></th></tr>${r.metrics.rows.map(x => `<tr><td>${KIND[x[0]][0]}</td><td>${esc(x[1])}</td><td>${esc(x[2])}</td><td>${x[3] ? "○ 一致" : "× 違う"}</td></tr>`).join("")}</table>`;
  simChart();
  simDraw();
}

function simDraw() {
  const r = STORE.res; if (!r || !document.getElementById("sim-3d")) return;
  const s = simSnap(), m = s.metrics, f = r.frames[STORE.i];
  document.getElementById("sim-clock").textContent = `${STORE.i} 秒 / ${r.frames.length - 1} 秒`;
  const tile = (name, v, sub) => `<div class="card"><div class="mute">${name}</div><div style="font-size:24px;font-weight:600">${v}</div><div class="mute" style="font-size:12px">${sub}</div></div>`;
  document.getElementById("sim-tiles").innerHTML = [
    tile("什器の再現 F1", pct(m.f1), `適合 ${pct(m.precision)} / 再現 ${pct(m.recall)}`),
    tile("見た什器の再現", pct(m.seen_recall), `見た ${m.seen_gt} / 全 ${m.gt} モジュール`),
    tile("什器の種類の正解", pct(m.kind_acc), `対応づいた ${m.matched} 個`),
    tile("位置の誤差", m.pos_err.toFixed(2) + " m", "モジュール中心の平均"),
    tile("主な商品 カテゴリ一致", pct(m.cat_acc), `1 枠ごとの認識は ${pct(r.named_acc)}（120 分類）`),
    tile("主な商品 大分類一致", pct(m.genre_acc), `7 分類 · ${m.faces} 面`),
    tile("ゴンドラのテーマ一致", STORE.cfg.themed == "0" ? "-" : pct(m.theme_acc), "面ごと · 17 テーマ"),
  ].join("");
  sim3d(s, f); simPlanDraw(s, f);
}

function simBox(pts) { let a = 1e9, b = -1e9; for (const p of pts) { a = Math.min(a, p); b = Math.max(b, p); } return [a, b]; }

// ---- 3D: 奥から順に面を塗るだけの簡単な描画（ライブラリを足さない）
function sim3d(s, f) {
  const cv = document.getElementById("sim-3d"), g = cv.getContext("2d"), { W, D, fixtures } = STORE.res.store;
  const cx = W / 2, cz = D / 2, dist = 1.7 * Math.max(W, D) / STORE.zoom, cy = Math.cos(STORE.yaw), sy = Math.sin(STORE.yaw), cp = Math.cos(STORE.pitch), sp = Math.sin(STORE.pitch);
  const project = (u, y, v) => {
    const x = u - cx, z = v - cz, X = x * cy + z * sy, Z0 = -x * sy + z * cy, Y = y * cp - Z0 * sp, Z = y * sp + Z0 * cp + dist, k = 760 / Math.max(Z, 0.1);
    return [cv.width / 2 + X * k, cv.height / 2 + 60 - Y * k, Z];
  };
  g.clearRect(0, 0, cv.width, cv.height);
  const line = cssVar("--line"), mute = cssVar("--mute");
  g.strokeStyle = line; g.lineWidth = 1;
  const poly = (ps) => { g.beginPath(); ps.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath(); };
  poly([project(0, 0, 0), project(W, 0, 0), project(W, 0, D), project(0, 0, D)]); g.stroke();
  const quads = [];
  for (const e of s.model) {
    const [c0, c1, c2, c3] = e.c, h = e.h, base = kindColor(e.kind);
    const lift = (c, y) => project(c[0], y, c[1]);
    quads.push({ ps:[lift(c0, h), lift(c1, h), lift(c2, h), lift(c3, h)], c:base, a:1 });
    for (const [a, b] of [[c0, c1], [c1, c2], [c2, c3], [c3, c0]]) quads.push({ ps:[lift(a, 0), lift(b, 0), lift(b, h), lift(a, h)], c:base, a:0.72 });
  }
  quads.forEach(q => q.z = q.ps.reduce((t, p) => t + p[2], 0) / 4);
  quads.sort((a, b) => b.z - a.z);
  for (const q of quads) { g.globalAlpha = q.a; g.fillStyle = q.c; poly(q.ps); g.fill(); g.globalAlpha = 1; g.strokeStyle = cssVar("--card"); g.stroke(); }
  g.setLineDash([4, 4]); g.strokeStyle = mute;
  for (const e of fixtures) {
    const lo = e.c - e.d / 2, hi = e.c + e.d / 2, [u0, v0, u1, v1] = e.axis == 0 ? [lo, e.a, hi, e.b] : [e.a, lo, e.b, hi], h = e.kind == "register" ? 1 : 1.8;
    for (const y of [0, h]) { poly([project(u0, y, v0), project(u1, y, v0), project(u1, y, v1), project(u0, y, v1)]); g.stroke(); }
  }
  g.setLineDash([]);
  const p = project(f.u, 1.3, f.v);
  g.fillStyle = cssVar("--fg"); g.beginPath(); g.arc(p[0], p[1], 6, 0, 7); g.fill();
}

function simPlanXf() {
  const cv = document.getElementById("sim-plan"), { W, D } = STORE.res.store, k = Math.min((cv.width - 24) / W, (cv.height - 24) / D);
  return { k, x:(u) => 12 + u * k, y:(v) => cv.height - 12 - v * k, u:(x) => (x - 12) / k, v:(y) => (cv.height - 12 - y) / k };
}

function simPlanDraw(s, f) {
  const cv = document.getElementById("sim-plan"), g = cv.getContext("2d"), r = STORE.res, { W, D, fixtures } = r.store, t = simPlanXf();
  g.clearRect(0, 0, cv.width, cv.height);
  g.strokeStyle = cssVar("--line"); g.lineWidth = 2; g.strokeRect(t.x(0), t.y(D), W * t.k, D * t.k);
  g.lineWidth = 1;
  for (const e of s.model) {
    g.beginPath(); e.c.forEach((c, i) => i ? g.lineTo(t.x(c[0]), t.y(c[1])) : g.moveTo(t.x(c[0]), t.y(c[1]))); g.closePath();
    g.fillStyle = kindColor(e.kind); g.globalAlpha = 0.8; g.fill(); g.globalAlpha = 1;
  }
  g.setLineDash([4, 3]); g.strokeStyle = cssVar("--mute");
  for (const e of fixtures) {
    const lo = e.c - e.d / 2, hi = e.c + e.d / 2, [u0, v0, u1, v1] = e.axis == 0 ? [lo, e.a, hi, e.b] : [e.a, lo, e.b, hi];
    g.strokeRect(t.x(u0), t.y(v1), (u1 - u0) * t.k, (v1 - v0) * t.k);
  }
  g.setLineDash([]);
  g.strokeStyle = cssVar("--fg"); g.lineWidth = 2; g.beginPath();
  r.frames.slice(0, STORE.i + 1).forEach((p, i) => i ? g.lineTo(t.x(p.u), t.y(p.v)) : g.moveTo(t.x(p.u), t.y(p.v))); g.stroke();
  // 今のカメラの向きと写る範囲（実機の水平画角 41.8 度）
  g.fillStyle = cssVar("--fg"); g.globalAlpha = 0.18; g.beginPath(); g.moveTo(t.x(f.u), t.y(f.v));
  for (const a of [-0.365, 0.365]) g.lineTo(t.x(f.u + 3.5 * Math.cos(f.h + a)), t.y(f.v + 3.5 * Math.sin(f.h + a)));
  g.closePath(); g.fill(); g.globalAlpha = 1;
  g.beginPath(); g.arc(t.x(f.u), t.y(f.v), 5, 0, 7); g.fill();
  if (STORE.draw) {
    g.strokeStyle = cssVar("--acc"); g.lineWidth = 2; g.beginPath();
    STORE.route.forEach((p, i) => i ? g.lineTo(t.x(p[0]), t.y(p[1])) : g.moveTo(t.x(p[0]), t.y(p[1]))); g.stroke();
    g.fillStyle = cssVar("--acc"); for (const p of STORE.route) { g.beginPath(); g.arc(t.x(p[0]), t.y(p[1]), 4, 0, 7); g.fill(); }
  }
}

function simPlan(cv) {
  const at = (e) => { const b = cv.getBoundingClientRect(), t = simPlanXf(); return [t.u((e.clientX - b.left) * cv.width / b.width), t.v((e.clientY - b.top) * cv.height / b.height)]; };
  cv.onclick = (e) => { if (STORE.draw && STORE.res) { STORE.route.push(at(e)); simDraw(); } };
  cv.onmousemove = (e) => {
    if (!STORE.res) return;
    const [u, v] = at(e);
    const hit = simSnap().model.find(m => { const [a0, a1] = simBox(m.c.map(c => c[0])), [b0, b1] = simBox(m.c.map(c => c[1])); return u >= a0 && u <= a1 && v >= b0 && v <= b1; });
    cv.title = hit ? `${KIND[hit.kind][0]}（${hit.n}×${hit.w} m）\n` + hit.mods.map((m, i) => `${i + 1}: ` + (m.faces.map(f => (f.theme ? `[${f.theme}] ` : "") + f.cats.slice(0, 2).map(c => `${c[0]} ${Math.round(c[1] * 100)}%`).join("・")).filter(Boolean).join(" ／ ") || "（名前なし）")).join("\n") : "";
  };
}

function simDrag(cv) {
  let last = null;
  cv.onpointerdown = (e) => { last = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = (e) => { if (!last) return; STORE.yaw += (e.clientX - last[0]) * 0.008; STORE.pitch = Math.min(1.5, Math.max(0.1, STORE.pitch + (e.clientY - last[1]) * 0.006)); last = [e.clientX, e.clientY]; simDraw(); };
  cv.onpointerup = () => last = null;
  cv.onwheel = (e) => { e.preventDefault(); STORE.zoom = Math.min(4, Math.max(0.4, STORE.zoom * (e.deltaY < 0 ? 1.1 : 0.9))); simDraw(); };
}

function simChart() {
  const el = document.getElementById("sim-chart"), snaps = STORE.res.snaps, W = 900, H = 230, L = 40, R = 90, T = 10, B = 28, tMax = snaps[snaps.length - 1].t || 1;
  const series = [["什器の再現 F1", "f1"], ["主な商品 大分類一致", "genre_acc"], ["主な商品 カテゴリ一致", "cat_acc"]], pal = PAL[scheme()];
  const x = (t) => L + (W - L - R) * t / tMax, y = (v) => T + (H - T - B) * (1 - v);
  const grid = [0, 0.5, 1].map(v => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("");
  const lines = series.map(([n, k], i) => {
    const last = snaps[snaps.length - 1].metrics[k];
    return `<path d="${snaps.map((s, j) => (j ? "L" : "M") + x(s.t).toFixed(1) + " " + y(s.metrics[k]).toFixed(1)).join("")}" fill="none" stroke="${pal[i]}" stroke-width="2" stroke-linejoin="round"/>
      <text x="${W - R + 6}" y="${y(last) + 4 + i * 2}">${pct(last)}</text>`;
  }).join("");
  el.innerHTML = `<div class="row" style="gap:14px;font-size:12px">${series.map(([n], i) => `<span><i style="display:inline-block;width:10px;height:2px;background:${pal[i]};vertical-align:middle;margin-right:4px"></i>${n}</span>`).join("")}</div>
    <div style="position:relative"><svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block" role="img" aria-label="時間ごとの再現率">
    ${grid}${lines}<text x="${L}" y="${H - 8}">0 秒</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${tMax} 秒</text>
    <line id="sim-cross" y1="${T}" y2="${H - B}" stroke="var(--mute)" visibility="hidden"/><rect id="sim-hit" x="${L}" y="0" width="${W - L - R}" height="${H}" fill="transparent"/></svg>
    <div id="sim-tip" class="card" style="position:absolute;display:none;pointer-events:none;font-size:12px;margin:0;padding:6px 8px"></div></div>`;
  const svg = el.querySelector("svg"), cross = el.querySelector("#sim-cross"), tip = el.querySelector("#sim-tip"), hit = el.querySelector("#sim-hit");
  hit.onmousemove = (e) => {
    const b = svg.getBoundingClientRect(), t = ((e.clientX - b.left) * W / b.width - L) / (W - L - R) * tMax;
    const s = snaps.reduce((a, c) => Math.abs(c.t - t) < Math.abs(a.t - t) ? c : a);
    cross.setAttribute("x1", x(s.t)); cross.setAttribute("x2", x(s.t)); cross.setAttribute("visibility", "visible");
    tip.style.display = "block"; tip.style.left = Math.min(e.clientX - b.left + 12, b.width - 200) + "px"; tip.style.top = "20px";
    tip.innerHTML = `<b>${s.t} 秒</b><br>` + series.map(([n, k]) => `${n} ${pct(s.metrics[k])}`).join("<br>");
  };
  hit.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); tip.style.display = "none"; };
}

function simBatch() {
  const b = STORE.batch, el = document.getElementById("sim-batch"); if (!el) return;
  if (!b || !b.rows) { el.textContent = "まだありません（python products/store_sim.py --batch）"; return; }
  document.getElementById("sim-bnote").textContent = `${b.seeds} 店の平均 · 検出は sku110k-2 の実測に合わせた · 分類の正解率 ${pct(b.acc)}: ${b.source} · 商品でない枠 ${pct(b.np)}（${b.np_n} 枠の Gemini）`;
  const acc = document.querySelector('[data-k="acc"]'); if (acc) acc.placeholder = b.acc;
  el.innerHTML = `<table><tr><th>条件</th><th class="num">F1</th><th class="num">最悪の店</th><th class="num">種類</th>${Object.values(KIND).map(k => `<th class="num">${k[0].replace("ゴンドラ", "").replace("冷蔵・冷凍", "").replace("什器", "")}</th>`).join("")}<th class="num">位置誤差</th><th class="num">カテゴリ</th><th class="num">大分類</th><th class="num">テーマ</th></tr>${b.rows.map(r => `<tr><td>${esc(r.name)}</td><td class="num">${pct(r.f1)}</td><td class="num">${pct(r.f1_min)}</td><td class="num">${pct(r.kind_acc)}</td>${Object.keys(KIND).map(k => `<td class="num">${pct(r.by_kind[k])}</td>`).join("")}<td class="num">${r.pos_err.toFixed(2)} m</td><td class="num">${pct(r.cat_acc)}</td><td class="num">${pct(r.genre_acc)}</td><td class="num">${pct(r.theme_acc)}</td></tr>`).join("")}</table>`;
}
