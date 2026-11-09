// 記録の再生（products/replay.html）。端末で記録したデータに、アプリと同じ認識と案内を動かした結果を時間に沿って見る
const R = { list:[], name:"", detector:"app", hz:1, data:null, i:0, timer:null, depth:false, img:new Image(), job:false };
const KIND = { island:"#3987e5", wall:"#d95926", case:"#199e70", end:"#c98500", register:"#d55181" };
const KIND_NAME = { island:"島ゴンドラ", wall:"壁面ゴンドラ", case:"冷蔵・冷凍ケース", end:"エンド什器", register:"レジ" };
const EVENT_NAME = { tap:"タップ", double_tap:"ダブルタップ", long_press:"長押し", speech:"聞き取り" };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]));
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (v) => v == null ? "-" : Math.round(v * 100) + "%";
const post = (u, b) => fetch(u, { method:"POST", body:JSON.stringify(b) }).then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); });

$("main").innerHTML = `
<div class="card row">
  <label>記録 <select id="name"></select></label>
  <button id="pull">端末から取り込む</button>
  <label>検出器 <select id="detector"><option value="app">今のアプリ</option><option value="sku110k-2">sku110k-2</option></select></label>
  <label>頻度 <select id="hz"><option value="1">1 回/秒（アプリと同じ）</option><option value="2">2 回/秒</option><option value="5">5 回/秒</option></select></label>
  <button id="run">認識を実行</button>
  <span class="mute" id="status"></span>
</div>
<div id="body" hidden>
  <div class="card row"><button id="play">▶</button><input id="t" type="range" min="0" value="0" style="flex:1"><span id="clock"></span></div>
  <div class="tiles" id="tiles"></div>
  <div class="cols">
    <div class="card"><div class="row"><b>カメラと認識</b><label class="mute small"><input type="checkbox" id="depth"> 深度を表示</label></div>
      <canvas id="cam" width="720" height="1280" style="max-height:640px;margin:6px auto 0"></canvas><div class="mute small" id="camnote"></div></div>
    <div class="card"><b>店モデル（上から見た図）</b><canvas id="plan" width="760" height="640" style="width:100%"></canvas>
      <div class="row small" id="legend" style="gap:10px"></div><div class="mute small" id="hover"></div></div>
    <div class="card"><b>ここでタップしたら</b><div class="speak" id="tap"></div>
      <div class="mute small" id="focus"></div>
      <div style="margin-top:12px"><b>長押しの質問（今の位置で AI に聞く）</b>
      <div class="row" style="margin-top:4px"><input type="text" id="q" value="カップ麺はどこにある？" style="flex:1"><button id="ask">聞く</button></div>
      <div id="answer" style="margin-top:6px;font-weight:600"></div></div>
      <div style="margin-top:12px"><b>記録中の操作</b><div id="events" class="small" style="margin-top:4px;max-height:220px;overflow:auto"></div></div></div>
  </div>
  <div class="card"><b>AI に渡す文脈（この瞬間）</b><pre class="log" id="context"></pre></div>
  <div class="card"><b>記述言語で書いた店モデル（最後）</b><pre class="log" id="text"></pre></div>
</div>`;

$("legend").innerHTML = Object.entries(KIND_NAME).map(([k, n]) => `<span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${KIND[k]};margin-right:4px"></i>${n}</span>`).join("");
$("name").onchange = (e) => { R.name = e.target.value; load(); };
$("detector").onchange = (e) => { R.detector = e.target.value; load(); };
$("hz").onchange = (e) => { R.hz = +e.target.value; };
$("pull").onclick = () => job("/api/replay/pull", {});
$("run").onclick = () => R.name && job("/api/replay/run", { name:R.name, detector:R.detector, hz:R.hz });
$("t").oninput = (e) => { R.i = +e.target.value; draw(); };
$("play").onclick = play;
$("depth").onchange = (e) => { R.depth = e.target.checked; draw(); };
$("ask").onclick = askAi;
R.img.onload = paintCam;
refresh();

async function refresh() {
  R.list = await (await fetch("/api/replay/list")).json();
  const s = $("name"); s.innerHTML = R.list.map(r => `<option value="${r.name}">${r.name}（${r.keyframes} 枚・動画 ${r.mp4_mb} MB${r.pulled ? "" : "・取り込み途中"}）</option>`).join("");
  if (!R.list.length) { $("status").textContent = "記録がありません。アプリの「● 記録」で歩いてから「端末から取り込む」を押してください"; return; }
  if (!R.list.some(r => r.name == R.name)) R.name = R.list[0].name;
  s.value = R.name; load();
}

async function load() {
  stopPlay();
  const r = await fetch(`/api/replay/data/${R.name}/${R.detector}`);
  if (!r.ok) { R.data = null; $("body").hidden = true; if (!R.job) $("status").textContent = "「認識を実行」を押すと、この記録で認識を動かします"; return; }
  R.data = await r.json(); R.i = 0;
  $("body").hidden = false; if (!R.job) $("status").textContent = "";
  $("t").max = R.data.steps.length - 1; $("t").value = 0;
  $("text").textContent = R.data.text;
  const s = R.data.summary;
  const tile = (n, v, sub) => `<div class="card"><div class="mute small">${n}</div><div class="big">${v}</div><div class="mute small">${sub}</div></div>`;
  $("tiles").innerHTML = [tile("長さ", s.seconds + " 秒", `${s.steps} ステップ`), tile("追跡できた割合", pct(s.tracked_ratio), `画像 ${s.keyframes} 枚・姿勢 ${s.poses}`), tile("検出（1 回あたり）", s.boxes_per_step, "枠の数"),
    tile("名前付き（1 回）", s.named_per_step, `置けた ${s.placed_per_step}`), tile("深度がある枠", pct(s.depth_ratio), "枠の中央の深度"), tile("処理時間", s.ms_per_step + " ms", "このPCの認識 1 回"),
    tile("地図の商品", s.items, `平面 ${s.planes}`), tile("推定した什器", s.fixtures, "最後の店モデル")].join("");
  $("events").innerHTML = R.data.events.length ? R.data.events.map(e => `<button class="ev" data-at="${e.at}">${e.at.toFixed(1)} 秒 ${EVENT_NAME[e.type] || e.type}${e.text ? "「" + esc(e.text) + "」" : ""}</button>`).join("") : "<span class='mute'>なし</span>";
  $("events").querySelectorAll("button").forEach(b => b.onclick = () => seek(+b.dataset.at, b.textContent.includes("聞き取り") ? b.textContent : null));
  draw();
}

function seek(at, speech) {
  const steps = R.data.steps; let best = 0;
  steps.forEach((s, k) => { if (Math.abs(s.t - at) < Math.abs(steps[best].t - at)) best = k; });
  R.i = best; $("t").value = best;
  if (speech) { const m = speech.match(/「(.*)」/); if (m) $("q").value = m[1]; }
  draw();
}

async function job(url, body) {
  try { await post(url, body); R.job = true; poll(); } catch (e) { $("status").textContent = e.message; }
}

async function poll() {
  const s = await (await fetch("/api/replay/status")).json();
  $("status").textContent = (s.msg || "") + (s.total ? `（${s.done}/${s.total}）` : "");
  if (s.state == "running" || s.busy) return setTimeout(poll, 1000);
  R.job = false;
  await refresh();
  $("status").textContent = s.msg || "";
}

function play() {
  if (R.timer) return stopPlay();
  if (R.i >= R.data.steps.length - 1) R.i = 0;
  $("play").textContent = "⏸";
  R.timer = setInterval(() => { if (R.i >= R.data.steps.length - 1) return stopPlay(); R.i++; $("t").value = R.i; draw(); }, Math.max(100, 1000 / R.data.hz / 2));
}
function stopPlay() { clearInterval(R.timer); R.timer = null; $("play").textContent = "▶"; }

const step = () => R.data.steps[R.i];
function lastModel() { for (let k = R.i; k >= 0; k--) if (R.data.steps[k].model) return R.data.steps[k].model; return []; }

function draw() {
  const s = step();
  $("clock").textContent = `${s.t.toFixed(1)} 秒 / ${R.data.summary.seconds} 秒${s.state != "TRACKING" ? "（追跡なし）" : ""}`;
  $("tap").textContent = s.tap ? s.tap.text : (s.state != "TRACKING" ? "追跡が止まっています（アプリは案内できません）" : "-");
  $("focus").textContent = s.focus ? `2 回たたくと: ${{ PRODUCT:"手前の商品を Cloud Vision で調べる", SHELF:"「商品を近づけて」と案内", STEP_BACK:"「商品を近づけて」と案内", NOTHING:"「商品が見えません」と案内" }[s.focus]}` : "";
  $("context").textContent = s.context || "（この瞬間の文脈はありません）";
  const src = R.depth && s.has_depth ? `/api/replay/depth/${R.name}/${s.n}.png` : `/api/replay/file/${R.name}/${s.image}`;
  if (R.img.dataset.src != src) { R.img.dataset.src = src; R.img.src = src; } else paintCam();
  planDraw();
}

function paintCam() {
  const c = $("cam"), g = c.getContext("2d"), s = step();
  if (R.img.naturalWidth && c.width != R.img.naturalWidth && !R.depth) { c.width = R.img.naturalWidth; c.height = R.img.naturalHeight; }
  g.drawImage(R.img, 0, 0, c.width, c.height);
  const dets = s.dets || [];
  g.lineWidth = 2; g.font = "22px system-ui,'Yu Gothic UI',sans-serif"; g.textBaseline = "top";
  for (const [x0, y0, x1, y1, , name] of dets) {
    g.strokeStyle = name ? "#39ff6a" : "#ffffff66";
    g.strokeRect(x0 * c.width, y0 * c.height, (x1 - x0) * c.width, (y1 - y0) * c.height);
    if (name) { const w = g.measureText(name).width + 6; g.fillStyle = "#0b3d1c"; g.fillRect(x0 * c.width, y0 * c.height, w, 26); g.fillStyle = "#fff"; g.fillText(name, x0 * c.width + 3, y0 * c.height + 2); }
  }
  const named = dets.filter(d => d[5]).length, depth = dets.filter(d => d[7] != null).length;
  $("camnote").textContent = dets.length ? `枠 ${dets.length} 個（緑 = 名前付き ${named}、白 = 名前なし）· 深度あり ${depth}` : "認識なし";
}

function planDraw() {
  const c = $("plan"), g = c.getContext("2d"), steps = R.data.steps.filter(s => s.cam);
  g.clearRect(0, 0, c.width, c.height);
  const xs = [], zs = [];
  for (const s of steps) { xs.push(s.cam[0]); zs.push(s.cam[2]); }
  for (const f of R.data.final_model) for (const p of f.c) { xs.push(p[0]); zs.push(p[1]); }
  if (!xs.length) return;
  const x0 = Math.min(...xs) - 1, x1 = Math.max(...xs) + 1, z0 = Math.min(...zs) - 1, z1 = Math.max(...zs) + 1, k = Math.min(c.width / (x1 - x0), c.height / (z1 - z0));
  const X = (x) => (x - x0) * k, Z = (z) => (z - z0) * k;
  for (const f of lastModel()) {
    g.beginPath(); f.c.forEach((p, i) => i ? g.lineTo(X(p[0]), Z(p[1])) : g.moveTo(X(p[0]), Z(p[1]))); g.closePath();
    g.fillStyle = KIND[f.kind]; g.globalAlpha = 0.8; g.fill(); g.globalAlpha = 1;
  }
  g.strokeStyle = css("--fg"); g.lineWidth = 2; g.beginPath();
  R.data.steps.slice(0, R.i + 1).filter(s => s.cam).forEach((s, i) => i ? g.lineTo(X(s.cam[0]), Z(s.cam[2])) : g.moveTo(X(s.cam[0]), Z(s.cam[2]))); g.stroke();
  const s = step();
  if (s.cam) {
    g.fillStyle = css("--fg");
    if (s.fwd) {
      g.globalAlpha = 0.18; g.beginPath(); g.moveTo(X(s.cam[0]), Z(s.cam[2]));
      const a = Math.atan2(s.fwd[1], s.fwd[0]);
      for (const d of [-0.365, 0.365]) g.lineTo(X(s.cam[0] + 3.5 * Math.cos(a + d)), Z(s.cam[2] + 3.5 * Math.sin(a + d)));
      g.closePath(); g.fill(); g.globalAlpha = 1;
    }
    g.beginPath(); g.arc(X(s.cam[0]), Z(s.cam[2]), 5, 0, 7); g.fill();
    if (s.tap && s.tap.at) { g.strokeStyle = css("--acc"); for (const p of s.tap.at) { g.beginPath(); g.arc(X(p[0]), Z(p[2]), 9, 0, 7); g.stroke(); } }
  }
  c.onmousemove = (e) => {
    const b = c.getBoundingClientRect(), x = (e.clientX - b.left) * c.width / b.width / k + x0, z = (e.clientY - b.top) * c.height / b.height / k + z0;
    const hit = lastModel().find(f => { const us = f.c.map(p => p[0]), vs = f.c.map(p => p[1]); return x >= Math.min(...us) && x <= Math.max(...us) && z >= Math.min(...vs) && z <= Math.max(...vs); });
    $("hover").textContent = hit ? `${KIND_NAME[hit.kind]}: ${hit.label}` : "什器に重ねると内容が出ます";
  };
}

async function askAi() {
  const out = $("answer"); out.textContent = "問い合わせ中…";
  try { const r = await post("/api/replay/ask", { name:R.name, detector:R.detector, step:R.i, question:$("q").value }); out.textContent = r.answer; }
  catch (e) { out.textContent = e.message; }
}
