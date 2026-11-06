const NS = "http://www.w3.org/2000/svg";
const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"];

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function lineChart(svg, series, opts) {
  clear(svg);
  const W = 640, H = 300, L = 48, R = 48, T = 16, B = 32;
  const xs = series.flatMap(s => s.points.map(p => p[0]));
  if (!xs.length) { el("text", {x: 20, y: 40, fill: "#6b7280"}, svg, "데이터 없음"); return; }
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const axes = {};
  series.forEach(s => {
    const ys = s.points.map(p => p[1]).filter(v => v !== null && isFinite(v));
    const a = axes[s.axis || "left"] || {min: Infinity, max: -Infinity};
    a.min = Math.min(a.min, ...ys, s.zero === false ? Infinity : 0); a.max = Math.max(a.max, ...ys);
    axes[s.axis || "left"] = a;
  });
  for (const k in axes) { if (axes[k].max === axes[k].min) axes[k].max = axes[k].min + 1; }
  const sx = v => L + (v - x0) / (x1 - x0) * (W - L - R);
  const sy = (v, ax) => { const a = axes[ax]; return H - B - (v - a.min) / (a.max - a.min) * (H - T - B); };
  el("line", {x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: "#9ca3af"}, svg);
  el("line", {x1: L, y1: T, x2: L, y2: H - B, stroke: "#9ca3af"}, svg);
  for (const ax in axes) {
    const x = ax === "left" ? L - 4 : W - R + 4;
    [axes[ax].min, (axes[ax].min + axes[ax].max) / 2, axes[ax].max].forEach(v =>
      el("text", {x, y: sy(v, ax) + 4, "font-size": 10, "text-anchor": ax === "left" ? "end" : "start", fill: "#6b7280"}, svg, (+v).toPrecision(3)));
  }
  const step = Math.max(1, Math.ceil((x1 - x0) / 10));
  for (let v = x0; v <= x1; v += step) el("text", {x: sx(v), y: H - B + 14, "font-size": 10, "text-anchor": "middle", fill: "#6b7280"}, svg, v);
  el("text", {x: (W) / 2, y: H - 4, "font-size": 11, "text-anchor": "middle", fill: "#6b7280"}, svg, (opts && opts.xlabel) || "round");
  series.forEach((s, i) => {
    const pts = s.points.filter(p => p[1] !== null && isFinite(p[1]));
    const d = pts.map((p, k) => (k ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1], s.axis || "left").toFixed(1)).join("");
    el("path", {d, fill: "none", stroke: s.color || COLORS[i % COLORS.length], "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "", opacity: s.opacity || 1}, svg);
    if (s.label) {
      const ly = T + 14 * i + 8;
      el("rect", {x: W - R - 120, y: ly - 8, width: 10, height: 3, fill: s.color || COLORS[i % COLORS.length]}, svg);
      el("text", {x: W - R - 106, y: ly - 3, "font-size": 11, fill: "#374151"}, svg, s.label);
    }
  });
}

function tree(svg, data) {
  clear(svg);
  const W = 640, H = 300;
  const edges = data.edges.map(e => e.eid);
  const byEdge = {};
  data.assignments.forEach(a => { if (a.eid >= 0) (byEdge[a.eid] = byEdge[a.eid] || []).push(a); });
  const root = {x: W / 2, y: 28};
  el("circle", {cx: root.x, cy: root.y, r: 14, fill: "#1d2330"}, svg);
  el("text", {x: root.x + 20, y: root.y + 4, "font-size": 11}, svg, "global");
  const lastRound = data.rounds.length ? data.rounds[data.rounds.length - 1] : {};
  const comm = {};
  data.edge_rounds.filter(r => r.round === data.assignment_round).forEach(r => comm[r.eid] = r.status);
  edges.forEach((eid, i) => {
    const ex = (i + 0.5) * W / edges.length, ey = 120;
    const ok = (comm[eid] || "ok") === "ok";
    el("line", {x1: root.x, y1: root.y, x2: ex, y2: ey, stroke: ok ? "#9ca3af" : "#dc2626", "stroke-dasharray": ok ? "" : "4 3"}, svg);
    el("rect", {x: ex - 16, y: ey - 12, width: 32, height: 24, rx: 4, fill: ok ? COLORS[i % COLORS.length] : "#fecaca"}, svg);
    el("text", {x: ex, y: ey + 4, "font-size": 11, "text-anchor": "middle", fill: "#fff"}, svg, "E" + eid);
    const cs = byEdge[eid] || [];
    cs.forEach((c, k) => {
      const span = W / edges.length;
      const cx = ex - span / 2 + (k + 0.5) * span / Math.max(cs.length, 1), cy = 230 + (k % 2) * 30;
      const color = {matched: "#16a34a", late: "#d97706", dropped: "#dc2626"}[c.status] || "#9ca3af";
      el("line", {x1: ex, y1: ey + 12, x2: cx, y2: cy - 8, stroke: "#d1d5db"}, svg);
      el("circle", {cx, cy, r: 8, fill: color}, svg);
      el("text", {x: cx, y: cy + 3, "font-size": 8, "text-anchor": "middle", fill: "#fff"}, svg, c.cid);
    });
  });
  const un = data.assignments.filter(a => a.eid < 0).map(a => a.cid);
  el("text", {x: 8, y: H - 6, "font-size": 11, fill: "#6b7280"}, svg, `round ${data.assignment_round} · 미매칭/불참: ${un.join(", ") || "-"} · acc ${lastRound.test_acc !== undefined ? lastRound.test_acc.toFixed(4) : "-"}`);
}

function table(container, head, rows, cellStyle) {
  const t = document.createElement("table");
  t.className = "grid";
  const thead = t.createTHead().insertRow();
  head.forEach(h => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
  const tb = t.createTBody();
  rows.forEach((r, i) => {
    const tr = tb.insertRow();
    r.forEach((v, j) => {
      const td = tr.insertCell();
      td.textContent = v;
      if (cellStyle) cellStyle(td, i, j, v);
    });
  });
  container.innerHTML = "";
  container.appendChild(t);
}

function heatColor(v, lo, hi, hue) {
  const t = hi > lo ? (v - lo) / (hi - lo) : 0;
  return `hsl(${hue || 217},80%,${96 - 50 * Math.max(0, Math.min(1, t))}%)`;
}

function render(data) {
  const run = data.run;
  document.getElementById("status").textContent = run.status;
  document.getElementById("status").className = "tag " + run.status;
  const rounds = data.rounds;
  const last = rounds[rounds.length - 1] || {};
  const kpis = [
    ["backend", run.config.backend], ["dataset", run.config.dataset || "-"],
    ["clients / edges", `${run.config["num-clients"]} / ${run.config["num-edges"]}`],
    ["round", `${Math.max(0, rounds.length - 1)} / ${run.config["global-rounds"]}`],
    ["test acc", last.test_acc !== undefined ? last.test_acc.toFixed(4) : "-"],
    ["matching", run.config.matching.mode], ["edge strategy", run.config["edge-strategy"]],
    ["virtual time (s)", (rounds.reduce((a, r) => a + (r.virtual_ms || 0), 0) / 1000).toFixed(1)],
  ];
  const k = document.getElementById("kpis");
  k.innerHTML = "";
  kpis.forEach(([a, b]) => { const d = document.createElement("div"); d.className = "kpi"; d.innerHTML = `<span>${a}</span><b>${b}</b>`; k.appendChild(d); });
  tree(document.getElementById("tree"), data);
  lineChart(document.getElementById("acc"), [
    {label: "test acc", points: rounds.map(r => [r.round, r.test_acc])},
    {label: "test loss", axis: "right", points: rounds.map(r => [r.round, r.test_loss]), color: "#dc2626", dash: "5 3"},
  ]);
  lineChart(document.getElementById("traffic"), [
    {label: "MB", points: rounds.filter(r => r.round > 0).map(r => [r.round, ((r.bytes_up || 0) + (r.bytes_down || 0)) / 1048576])},
    {label: "virtual s", axis: "right", points: rounds.filter(r => r.round > 0).map(r => [r.round, (r.virtual_ms || 0) / 1000]), color: "#16a34a"},
  ]);
  if (data.matching) {
    const u = data.matching.utility, a = data.matching.assign;
    const flat = u.flat();
    const lo = Math.min(...flat), hi = Math.max(...flat);
    table(document.getElementById("matrix"), ["client"].concat(data.edges.map(e => "E" + e.eid)),
      u.map((row, j) => [j].concat(row.map((v, i) => (a[j] === i ? "● " : "") + v.toFixed(3)))),
      (td, i, j, v) => { if (j > 0) td.style.background = heatColor(u[i][j - 1], lo, hi); });
    const ranks = data.matching.priority_rank;
    const depth = Math.max(...ranks.map(r => r.length), 0);
    table(document.getElementById("rank"), ["rank"].concat(data.edges.map(e => "E" + e.eid)),
      Array.from({length: depth}, (_, r) => [r + 1].concat(ranks.map(x => x[r] !== undefined ? "C" + x[r] : ""))));
  }
  const rmax = Math.max(...data.edge_rounds.map(r => r.round), 0);
  table(document.getElementById("comm"), ["round", "edge", "status", "clients", "down MB", "up MB", "wall s", "virtual s"],
    data.edge_rounds.filter(r => r.round > rmax - 6).map(r => [r.round, "E" + r.eid, r.status, r.clients, (r.bytes_down / 1048576).toFixed(2), (r.bytes_up / 1048576).toFixed(2), r.seconds.toFixed(2), (r.virtual_ms / 1000).toFixed(2)]),
    (td, i, j, v) => { if (j === 2 && v !== "ok") td.style.background = "#fee2e2"; });
  const ent = data.clients.map(c => c.entropy), rat = data.clients.map(c => c.data_ratio);
  const allCounts = data.clients.flatMap(c => c.label_counts);
  const cmax = Math.max(...allCounts, 1);
  table(document.getElementById("heat"), ["client", "role", "R", "H"].concat([...Array(10).keys()].map(i => "y" + i)),
    data.clients.map(c => [c.cid, c.role, c.data_ratio.toFixed(3), c.entropy.toFixed(2)].concat(c.label_counts)),
    (td, i, j) => {
      if (j === 2) td.style.background = heatColor(rat[i], Math.min(...rat), Math.max(...rat), 140);
      if (j === 3) td.style.background = heatColor(ent[i], Math.min(...ent), Math.max(...ent), 25);
      if (j > 3) td.style.background = heatColor(data.clients[i].label_counts[j - 4], 0, cmax);
      if (j === 1 && data.clients[i].role !== "benign") td.style.color = "#dc2626";
    });
  const trend = data.client_trend;
  const flagged = new Set(data.outliers.map(o => +o.cid));
  lineChart(document.getElementById("trend"), Object.keys(trend).map((cid, i) => ({
    points: trend[cid].map(p => [p.round, p.loss]),
    color: flagged.has(+cid) ? "#dc2626" : "#94a3b8", width: flagged.has(+cid) ? 2.5 : 1, opacity: flagged.has(+cid) ? 1 : 0.6,
  })));
  document.getElementById("outliers").textContent = data.outliers.length ?
    "이상치(품질 q < 0.35): " + data.outliers.map(o => `C${o.cid} (q=${o.min_q.toFixed(2)})`).join(", ") : "이상치 없음";
  const hist = {};
  data.status_history.forEach(s => { (hist[s.cid] = hist[s.cid] || {})[s.round] = s; });
  const R = [...new Set(data.status_history.map(s => s.round))].sort((a, b) => a - b);
  table(document.getElementById("participation"), ["client"].concat(R.map(r => "R" + r)),
    Object.keys(hist).map(c => [c].concat(R.map(r => hist[c][r] ? (hist[c][r].eid >= 0 ? "E" + hist[c][r].eid : "-") : ""))),
    (td, i, j) => { if (j > 0) { const s = hist[Object.keys(hist)[i]][R[j - 1]]; if (s) td.className = "st-" + s.status; } });
  const ev = document.getElementById("events");
  ev.innerHTML = "";
  data.events.forEach(e => { const d = document.createElement("div"); d.className = "lv-" + e.level; d.textContent = `[R${e.round}] ${e.level} ${e.source}: ${e.message}`; ev.appendChild(d); });
  return run.status;
}

async function refresh() {
  try {
    const r = await fetch(`/api/run/${RUN_ID}`);
    const status = render(await r.json());
    if (status === "running") setTimeout(refresh, 3000);
  } catch (e) { setTimeout(refresh, 5000); }
}

refresh();
