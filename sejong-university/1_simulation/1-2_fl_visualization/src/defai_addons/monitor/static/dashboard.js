const S = { runId: null, snap: null, defaults: null, groupId: null, group: null,
  jobId: null, job: null, hyperFocus: 'server', socioPos: {}, sel: null,
  view: { facet: 'profit', malgo: 'cmp', mbar: 'sum', sim: 'dendro', contrib: 'pareto', res: 'cpu', ldfl: 'f1', gtrend: 'trend' } };
const $ = id => document.getElementById(id);
const api = (url, body) => fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {})
  .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

const ENC = {
  client: ['#e8892b', '#3f9b3f', '#9b59b6', '#e9c23a', '#8c5a3c', '#f29bb8', '#3aa39b', '#a3a32c', '#a2477a', '#8fd17a',
    '#c9a27e', '#d65ca3', '#1f6f6a', '#ffbf80', '#5e3c99', '#f3dc8a', '#2e6b2e', '#f7c6d6', '#c8a8dc', '#93cfc6'],
  ink: '#1b1b1b',
  edgeSym: [d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond, d3.symbolCross, d3.symbolWye],
  edgeGlyph: ['■', '▲', '◆', '✚', 'Y'],
  edgeDash: [null, '6 3', '1.5 2.5', '8 3 2 3', '3 2 1 2'],
  status: { ok: '#cfcfca', timeout: '#55554f', idle: '#ffffff', outlier: '#d03b3b' },
  alert: '#d03b3b',
  alg: { fedavg: { dash: null, marker: d3.symbolCircle, bar: null }, fedprox: { dash: '6 3', marker: d3.symbolSquare, bar: 'hatch' },
    qfedavg: { dash: '1.5 2.5', marker: d3.symbolTriangle, bar: 'outline' } },
};
const clientColor = i => ENC.client[((i % ENC.client.length) + ENC.client.length) % ENC.client.length];
const edgeSym = j => ENC.edgeSym[j % ENC.edgeSym.length];
const edgeGlyph = j => (j == null || j < 0) ? '·' : ENC.edgeGlyph[j % ENC.edgeGlyph.length];
const edgeDash = j => ENC.edgeDash[j % ENC.edgeDash.length];
const edgeLabel = j => (j == null || j < 0) ? '미배정' : `${edgeGlyph(j)}e${j}`;
const STATUS_KO = { ok: '정상', timeout: '타임아웃', idle: '미참여', outlier: '이상치', degraded: '저하', down: '중단' };
const statusOf = c => c.outlier ? 'outlier' : (c.status || 'ok');
const ALG_ORDER = { fedavg: 0, fedprox: 1, qfedavg: 2 };
const ALG_NAME = { fedavg: 'FedAvg', fedprox: 'FedProx', qfedavg: 'q-FedAvg' };
const ENV_SHORT = { ld_independent: '지역·독립', ld_sync: '지역·동기화', ld_central: '지역·중앙', fl_fedavg: '연합 FedAvg', fl_fedprox: '연합 FedProx', fl_fedavg_dp: '연합 FedAvg+DP' };
const ENV_DASH = [null, '6 3', '1.5 2.5', '8 3 2 3', '3 2 1 2', '10 3'];
const badge = (st, txt) => `<span class="badge st-${st}"><i></i>${txt || STATUS_KO[st] || st}</span>`;
const STATUS_TXT = { running: '실행 중', finished: '완료', error: '오류', created: '대기' };
const cross = (g, r, color) => g.append('path').attr('d', `M${-r},${-r}L${r},${r}M${r},${-r}L${-r},${r}`).attr('stroke', color || ENC.ink).attr('stroke-width', 1.6).attr('pointer-events', 'none');
function clientDot(g, i, r, st, malicious) {
  const col = clientColor(i);
  g.append('circle').attr('r', r).attr('fill', st === 'idle' ? 'var(--surface)' : col).attr('stroke', st === 'idle' ? col : 'var(--surface)').attr('stroke-width', st === 'idle' ? 1.5 : 0.8);
  if (st === 'timeout') g.append('circle').attr('r', r + 2.2).attr('fill', 'none').attr('stroke', ENC.status.timeout).attr('stroke-width', 1.4).attr('stroke-dasharray', '2 1.5');
  if (st === 'outlier') g.append('circle').attr('r', r + 2.2).attr('fill', 'none').attr('stroke', ENC.alert).attr('stroke-width', 1.8);
  if (malicious) cross(g, r * 0.7, ENC.ink);
}
function edgeMark(g, j, size) { g.append('path').attr('d', d3.symbol(edgeSym(j), size || 70)()).attr('fill', ENC.ink); }
function serverMark(g, size) { g.append('path').attr('d', d3.symbol(d3.symbolStar, size || 110)()).attr('fill', ENC.ink); }

function selSet() {
  if (!S.sel || !S.snap || !S.snap.rounds.length) return null;
  if (S.sel.kind === 'client') return new Set([S.sel.id]);
  const a = lastAssign();
  return new Set(a.map((e, i) => e === S.sel.id ? i : -1).filter(i => i >= 0));
}
const fadeOf = on => i => (!on || on.has(i)) ? 1 : 0.12;
const edgeOn = j => !S.sel || (S.sel.kind === 'edge' ? S.sel.id === j : lastAssign()[S.sel.id] === j);
function select(kind, id) {
  S.sel = (S.sel && S.sel.kind === kind && S.sel.id === id) ? null : { kind, id };
  renderSelChip();
  render();
}
function renderSelChip() {
  const el = $('selChip');
  if (!S.sel || !S.snap) { el.innerHTML = '<span class="hint">클라이언트·엣지를 클릭하면 모든 시각화에서 강조</span>'; return; }
  if (S.sel.kind === 'client') {
    const c = S.snap.clients[S.sel.id] || {}, e = lastAssign()[S.sel.id];
    el.innerHTML = `<span class="chip"><svg class="lg" width="11" height="11"><circle cx="5.5" cy="5.5" r="5" fill="${clientColor(S.sel.id)}"/></svg>선택: c${S.sel.id} · 숫자 ${(c.classes || []).join(',')} · 소속 ${edgeLabel(e)}${c.malicious ? ' · 악성 설정' : ''} <b id="selClear">✕</b></span>`;
  } else {
    el.innerHTML = `<span class="chip">선택: ${edgeLabel(S.sel.id)} 과 소속 클라이언트 ${selSet().size}명 <b id="selClear">✕</b></span>`;
  }
  $('selClear').onclick = () => { S.sel = null; renderSelChip(); render(); };
}
const SOURCE = {
  treeChart: ['★■● 구조', '서버 · 엣지 · 클라이언트 전체 구조'], rankChart: ['● 클라', '클라이언트'], matchTrendChart: ['■ 엣지', '엣지 서버 단위'],
  matchOutlierChart: ['● 엣지 안', '엣지 안의 클라이언트'], hyperChart: ['★■● 링크', '서버↔엣지↔클라이언트 통신 링크'], heatChart: ['● 데이터', '클라이언트 로컬 데이터'],
  facetChart: ['●↔■ 매칭', '클라이언트 ↔ 엣지 매칭'], pixelChart: ['●■ 링크', '클라이언트→엣지, 엣지→서버 링크'], parallelChart: ['● 링크', '클라이언트 통신 링크'],
  clientCmpChart: ['● 클라', '클라이언트 업데이트·학습'], gtrendChart: ['★ 전역', '서버의 전역 모델'], monAlgoCurve: ['★ 비교', '전역 모델 · 알고리즘별 실행 비교'],
  monAlgoBars: ['★ 비교', '전역 모델 · 알고리즘별 실행 비교'], simChart: ['● 업데이트', '클라이언트 모델 업데이트'], resChart: ['★ 서버', '서버 프로세스 자원'],
  logChart: ['● 클라', '클라이언트 로그'], sankeyChart: ['●→■→★', '클라이언트→엣지→서버 전송량'], contribChart: ['●→★', '클라이언트가 전역 모델에 반영된 기여'], ldflChart: ['별도 실험', '지역분산 vs 연합 비교 실험'],
};
const srcTag = id => `<span class="src" title="데이터 출처: ${SOURCE[id][1]}">${SOURCE[id][0]}</span> `;

function renderLegendStrip() {
  const n = S.snap ? S.snap.clients.length : 20, ne = S.snap ? S.snap.topology.edges.length : 3;
  const cl = d3.range(n).map(i => `<span class="cc" data-c="${i}" title="client ${i}" style="background:${clientColor(i)}">${i}</span>`).join('');
  const sw = V.swatch;
  $('legendStrip').innerHTML = `
    <div class="lgrow"><b>클라이언트</b>${cl}</div>
    <div class="lgrow"><b>구조</b>${sw({ type: 'sym', sym: d3.symbolStar })}서버 ${d3.range(ne).map(j => sw({ type: 'sym', sym: edgeSym(j) }) + 'e' + j).join(' ')}
      <b>상태</b>${sw({ color: ENC.status.ok })}정상 ${sw({ color: ENC.status.timeout })}타임아웃 ${sw({ color: ENC.status.idle, border: true })}미참여
      ${sw({ type: 'ring', color: ENC.alert })}이상치 ${sw({ type: 'cross' })}악성 설정
      <b>값</b><span class="grad"></span>낮음→높음
      <b>알고리즘</b>${['fedavg', 'fedprox', 'qfedavg'].map(a => sw({ type: 'ln', dash: ENC.alg[a].dash, marker: ENC.alg[a].marker }) + ALG_NAME[a]).join(' ')}</div>`;
  $('legendStrip').querySelectorAll('.cc').forEach(e => e.onclick = () => select('client', +e.dataset.c));
}

function fillForm(form, cfg) {
  [...form.elements].forEach(el => {
    if (!el.name || !(el.name in cfg)) return;
    if (el.type === 'checkbox') el.checked = !!cfg[el.name];
    else el.value = cfg[el.name] == null ? '' : cfg[el.name];
  });
}
function readForm(form) {
  const out = {};
  [...form.elements].forEach(el => {
    if (!el.name) return;
    if (el.type === 'checkbox') out[el.name] = el.checked;
    else if (el.type === 'number') out[el.name] = el.value === '' ? null : Number(el.value);
    else out[el.name] = el.value;
  });
  return out;
}

async function init() {
  S.defaults = await api('/api/defaults');
  fillForm($('simForm'), S.defaults.simulation);
  fillForm($('cmpForm'), S.defaults.comparison);
  $('envChecks').innerHTML = Object.entries(S.defaults.env_labels).map(([k, v]) =>
    `<label title="${v}"><input type="checkbox" data-env="${k}" ${S.defaults.comparison.envs.includes(k) ? 'checked' : ''}>${ENV_SHORT[k] || v}</label>`).join('');

  document.querySelectorAll('main .card').forEach(card => {
    const h = card.querySelector('.ch h2'), cb = card.querySelector('.cb');
    h.title = h.textContent;  // 좁은 화면에서 잘린 제목은 마우스로 확인
    const cap = card.querySelector('p.d');
    if (cb && cap && SOURCE[cb.id]) cap.insertAdjacentHTML('afterbegin', srcTag(cb.id));
  });
  document.querySelectorAll('.seg').forEach(seg => seg.onclick = e => {
    const b = e.target.closest('button'); if (!b) return;
    seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    S.view[seg.dataset.view] = b.dataset.v; render();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && S.sel) { S.sel = null; renderSelChip(); render(); } });
  $('drawerBtn').onclick = () => $('drawer').classList.toggle('open');
  $('drawerClose').onclick = () => $('drawer').classList.remove('open');
  $('runSelect').onchange = () => { S.runId = $('runSelect').value; S.snap = null; S.sel = null; S.socioPos = {}; S.hyperFocus = 'server'; loadRun(); };
  const start = async () => {
    const m = await api('/api/runs', { config: readForm($('simForm')) });
    S.runId = m.id; S.snap = null; S.sel = null; S.socioPos = {}; $('drawer').classList.remove('open');
    await refreshRuns(); loadRun();
  };
  $('runBtn').onclick = start;
  $('runBtn2').onclick = start;
  $('cmpBtn').onclick = async () => {
    const r = await api('/api/runs/compare', { config: readForm($('simForm')), algorithms: ['fedavg', 'fedprox', 'qfedavg'],
      seeds: Number($('seedCount').value) || 1 });
    S.groupId = r.group; S.group = null; $('drawer').classList.remove('open'); await refreshRuns(); loadGroup();
  };
  $('stopBtn').onclick = () => S.runId && api(`/api/runs/${S.runId}/stop`, {});
  $('saveBtn').onclick = async () => { if (!S.runId) return; const r = await api(`/api/runs/${S.runId}/save`, {}); alert('저장됨: ' + r.path); };
  $('groupSel').onchange = () => { S.groupId = $('groupSel').value; S._userGroup = true; S.group = null; loadGroup(); };
  $('cmpMetric').onchange = render;
  $('ldflMetric').onchange = () => renderLdflCard($('ldflChart'));
  $('jobSel').onchange = () => { S.jobId = $('jobSel').value; loadJob(); };
  $('cmpRunBtn').onclick = async () => {
    const cfg = readForm($('cmpForm'));
    cfg.envs = [...document.querySelectorAll('#envChecks input:checked')].map(x => x.dataset.env);
    const r = await api('/api/comparisons', { config: cfg });
    S.jobId = r.id; S.job = null; $('drawer').classList.remove('open'); await refreshJobs(); loadJob();
  };
  window.addEventListener('resize', () => { clearTimeout(S._rs); S._rs = setTimeout(render, 200); });

  renderLegendStrip(); renderSelChip();
  await refreshRuns(); await refreshJobs();
  if (S.runId) loadRun(); else render();
  if (S.groupId) loadGroup();
  if (S.jobId) loadJob();
  setInterval(tick, 1000);
  setInterval(refreshRuns, 4000);
}

async function tick() {
  if (S.snap && S.snap.status === 'running') loadRun();
  if (S.group && S.group.runs.some(r => r.status === 'running' || r.status === 'created')) loadGroup();
  if (S.job && (S.job.status === 'running' || S.job.status === 'created')) loadJob();
}

async function refreshRuns() {
  let runs;
  try { runs = await api('/api/runs'); } catch (e) { return; }
  const sel = $('runSelect');
  sel.innerHTML = runs.length ? runs.map(r => `<option value="${r.id}">${r.name && r.name.includes('·') ? r.name : (ALG_NAME[r.algorithm] || r.name)} · ${r.id} · ${r.n_rounds}/${r.total_rounds || '?'} · ${STATUS_TXT[r.status] || r.status}</option>`).join('')
    : '<option value="">실행 없음 — ▶ 실행</option>';
  if (!S.runId && runs.length) S.runId = runs[0].id;
  if (S.runId) sel.value = S.runId;
  const groups = [...new Set(runs.filter(r => r.group).map(r => r.group))];
  const gs = $('groupSel');
  gs.innerHTML = groups.length ? groups.map(g => {
    const n = runs.filter(r => r.group === g), done = n.filter(r => r.status === 'finished').length;
    return `<option value="${g}">${g} · ${done}/${n.length} 완료</option>`;
  }).join('') : '<option value="">(없음)</option>';
  if (!S.groupId && groups.length) S.groupId = groups[0];
  if (S.groupId) gs.value = S.groupId;
}

async function loadRun() {
  if (!S.runId || S._loadingId === S.runId) return;  // 같은 실행의 폴링이 겹치지 않게
  const id = S.runId;
  S._loadingId = id;
  let snap;
  try { snap = await api(`/api/runs/${id}`); } catch (e) { return; } finally { if (S._loadingId === id) S._loadingId = null; }
  if (id !== S.runId) return;  // 응답 도중 다른 실행을 선택한 경우 오래된 응답은 버린다
  const nChanged = !S.snap || S.snap.clients.length !== snap.clients.length || S.snap.topology.edges?.length !== snap.topology.edges?.length;
  S.snap = snap;
  const st = snap.status;
  $('runStatus').innerHTML = badge(st, STATUS_TXT[st] || st);
  $('stopBtn').disabled = st !== 'running';
  $('saveBtn').disabled = false;
  if (nChanged) renderLegendStrip();
  const g = snap.meta && snap.meta.group;
  if (g && g !== S.groupId && !S._userGroup) { S.groupId = g; S.group = null; $('groupSel').value = g; loadGroup(); }
  renderSelChip();
  render();
}

async function loadGroup() {
  if (!S.groupId || S._groupLoading) return;
  S._groupLoading = true;
  const gid = S.groupId;
  try { S.group = await api(`/api/groups/${gid}`); } catch (e) { S.group = null; } finally { S._groupLoading = false; }
  if (gid !== S.groupId) return;
  safe(() => renderAlgoCards(S.snap && S.snap.views));
}

async function refreshJobs() {
  const jobs = await api('/api/comparisons');
  $('jobSel').innerHTML = jobs.length ? jobs.map(j => `<option value="${j.id}">${j.id} · ${STATUS_TXT[j.status] || j.status}</option>`).join('') : '<option value="">(없음)</option>';
  if (!S.jobId && jobs.length) S.jobId = jobs[0].id;
  if (S.jobId) $('jobSel').value = S.jobId;
}

async function loadJob() {
  if (!S.jobId) return;
  S.job = await api(`/api/comparisons/${S.jobId}`);
  $('jobStatus').innerHTML = badge(S.job.status === 'finished' ? 'finished' : S.job.status, STATUS_TXT[S.job.status]);
  if (S.job.status !== 'running') refreshJobs();
  safe(() => renderLdflCard($('ldflChart')));
}

const safe = fn => { try { fn(); } catch (e) { console.error(e); } };
const lastAssign = () => {
  const r = S.snap.rounds; if (!r.length) return [];
  return r[r.length - 1].clients.map(c => c.edge);
};
const emptyAll = (ids, msg) => ids.forEach(id => V.empty($(id), msg));
const RUN_IDS = ['treeChart', 'rankChart', 'matchTrendChart', 'matchOutlierChart', 'hyperChart', 'heatChart', 'facetChart', 'pixelChart',
  'parallelChart', 'clientCmpChart', 'gtrendChart', 'simChart', 'resChart', 'logChart', 'sankeyChart', 'contribChart'];

function render() {
  const v = S.snap && S.snap.views;
  renderRunCards(v);
  safe(() => renderAlgoCards(v));
  safe(() => renderLdflCard($('ldflChart')));
}

function renderRunCards(v) {
  if (!v || !v.trend) {
    $('kpis').innerHTML = S.snap && S.snap.error ? `<div class="kpi"><pre class="err">${S.snap.error}</pre></div>` : '';
    emptyAll(RUN_IDS, S.snap && S.snap.error ? '오류' : (S.snap ? '첫 라운드 대기 중…' : '▶ 실행 버튼으로 시뮬레이션을 시작하세요'));
    return;
  }
  const on = selSet();
  safe(() => renderKpis(v));
  safe(() => renderTree(v, $('treeChart'), on));
  safe(() => renderRank(v, $('rankChart'), on));
  safe(() => renderMatchTrend($('matchTrendChart')));
  safe(() => renderMatchOutliers($('matchOutlierChart'), on));
  safe(() => renderHyper(v, $('hyperChart'), on));
  safe(() => renderHeat(v.data, $('heatChart'), on));
  safe(() => renderFacets($('facetChart'), on));
  safe(() => renderPixels($('pixelChart'), on));
  safe(() => renderParallel($('parallelChart'), on));
  safe(() => renderClientCmp(v, $('clientCmpChart'), on));
  safe(() => S.view.gtrend === 'digit' ? renderClassAcc($('gtrendChart')) : renderGtrend(v, $('gtrendChart')));
  safe(() => {
    if (!v.similarity || !v.similarity.tree) V.empty($('simChart'), '업데이트 스케치 부족');
    else if (S.view.sim === 'socio') renderSocio(v.similarity, $('simChart'), on); else renderDendro(v.similarity, $('simChart'), on);
  });
  safe(() => renderResource(v, $('resChart')));
  safe(() => renderSparkTable(v, $('logChart'), on));
  safe(() => renderSankey(v, $('sankeyChart'), on));
  safe(() => S.view.contrib === 'stream' ? renderStream(v, $('contribChart'), on) : renderPareto(v, $('contribChart'), on));
}

function renderAlgoCards(v) {
  const g = S.group, ce = $('monAlgoCurve'), be = $('monAlgoBars');
  if (!g || !g.runs.length) {
    $('monAlgoInfo').innerHTML = srcTag('monAlgoCurve') + '⚙ 설정 · 실행 → ⇄ 비교 실행으로 세 알고리즘을 같은 설정에서 비교';
    V.empty(ce, '비교 그룹 없음'); V.empty(be, '비교 그룹 없음'); return;
  }
  const A = groupByAlg(g.runs), seeds = [...new Set(g.runs.map(r => r.seed))];
  const done = g.runs.filter(r => r.status === 'finished').length;
  $('monAlgoInfo').innerHTML = srcTag('monAlgoCurve') + (S.view.malgo === 'delta'
    ? `FedAvg 대비 차이(같은 seed, 0 위 = 우수) → 개선이 끝까지 유지되는지 · seed ${seeds.length}`
    : `세 알고리즘 곡선(띠 = seed 범위) → 더 빨리·높게 수렴하는 쪽 · seed ${seeds.length} · ${done}/${g.runs.length} 완료`);
  const curV = v && S.snap && S.snap.meta && !S.snap.meta.group ? v : null;
  if (S.view.malgo === 'delta') algoDelta(ce, A, g); else algoCurves(ce, A, g, curV);
  if (S.view.mbar === 'digit') algoDigitBars(be, A);
  else {
    const M = [['최종 정확도', r => r.final.accuracy], ['하위10% 클라', r => r.final.fair_worst10], ['최저 숫자', r => r.class_acc ? d3.min(r.class_acc) : null]];
    V.bars(be, { groups: M.map(m => m[0]), keys: A.map(a => a.name), colors: A.map(() => ENC.ink), styles: A.map(a => a.bar), yDomain: [0, 1], yFmt: d3.format('.2f'),
      values: M.map(([, fn]) => A.map(a => mstd(a.runs.map(fn))?.m ?? null)) });
  }
}

function alertsOf(v) {
  const trend = v.trend.alerts.map(a => ({ round: a.round, kind: '추세', msg: a.msg, lvl: 'alert' }));
  const logs = S.snap.logs.filter(l => l.level === 'alert' || l.event === 'timeout').map(l => ({
    round: l.round, kind: l.event, lvl: l.level === 'alert' ? 'alert' : 'warn',
    msg: `client ${l.client}${l.z != null ? ' · z=' + l.z : ''}${l.latency_ms != null ? ' · ' + l.latency_ms + 'ms' : ''}${l.excluded ? ' · 집계 제외' : ''}` }));
  return trend.concat(logs).sort((a, b) => a.round - b.round);
}

function renderKpis(v) {
  const t = v.trend, n = t.round.length, rs = S.snap.rounds, last = rs[rs.length - 1];
  const totalMB = d3.sum(rs, r => r.bytes_total) / 2 ** 20;
  const nOut = d3.sum(rs, r => r.clients.filter(c => c.outlier).length);
  const nTo = d3.sum(rs, r => r.clients.filter(c => c.status === 'timeout').length);
  const dAcc = n > 1 ? t.accuracy[n - 1] - t.accuracy[n - 2] : 0;
  const ca = last.class_acc;
  const kp = [
    ['라운드', `${n} / ${S.snap.config.rounds}`, `${ALG_NAME[S.snap.config.algorithm] || ''} · 엣지 ${S.snap.topology.edges.length} · 클라이언트 ${S.snap.clients.length}`],
    ['전역 정확도', V.fmt.pct(t.accuracy[n - 1]), `직전 대비 ${dAcc >= 0 ? '+' : ''}${(dAcc * 100).toFixed(1)}%p · F1 ${V.fmt.f3(t.f1[n - 1])}`],
    ['최저 숫자 정확도', ca ? V.fmt.pct(d3.min(ca)) : '–', ca ? `숫자 ${ca.indexOf(d3.min(ca))} · 하위10% 클라이언트 ${V.fmt.pct(t.fair_worst[n - 1])}` : ''],
    ['누적 통신량', V.fmt.f1(totalMB) + ' MB', `모델 ${(S.snap.topology.model_bytes / 1024).toFixed(0)} KB · 참여 ${last.participants}`],
    ['이상치 · 타임아웃', `${nOut} · ${nTo}`, `악성 설정 ${S.snap.clients.filter(c => c.malicious).length}명 · 방어 ${S.snap.config.defense ? 'ON' : 'OFF'}`],
  ];
  const al = alertsOf(v), latest = al[al.length - 1];
  $('kpis').innerHTML = kp.map(([k, val, d]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${val}</div><div class="d" title="${d}">${d}</div></div>`).join('') +
    `<div class="kpi alerts" id="alertKpi"><div class="k">경보 (마우스 → 최근 목록)</div>
      <div class="v">${al.length ? badge('alert', al.length + '건') : badge('finished', '없음')}</div>
      <div class="d">${latest ? `R${latest.round} · ${latest.kind} · ${latest.msg}` : '추세 급락·발산·정체, 이상치, 타임아웃 없음'}</div></div>`;
  $('alertKpi').onmousemove = e => V.tip(e, `최근 경보 ${Math.min(14, al.length)} / ${al.length}건`,
    al.slice(-14).reverse().map(a => [`R${a.round} · ${a.kind}`, a.msg, a.lvl === 'alert' ? ENC.alert : ENC.status.timeout]));
  $('alertKpi').onmouseleave = V.untip;
}

function renderClassAcc(el) {
  el = V.clear(el);
  const rs = S.snap.rounds.filter(r => r.class_acc);
  if (!rs.length) return V.empty(el, '숫자별 정확도 기록 없음');
  const K = rs[0].class_acc.length;
  const owners = d3.range(K).map(k => S.snap.clients.filter(c => (c.class_counts || [])[k] > 0).length);
  const f = V.frame(el, null, { l: 42, r: 64, b: 34, t: 2 });
  const rh = f.h / K;
  const x = d3.scaleBand().domain(rs.map(r => r.round)).range([0, f.w]).padding(0.04);
  const cs = V.seq([0, 1]);
  rs.forEach(r => r.class_acc.forEach((a, k) => f.g.append('rect').attr('x', x(r.round)).attr('y', k * rh).attr('width', x.bandwidth())
    .attr('height', Math.max(1, rh - 1)).attr('fill', cs(a))
    .on('mousemove', e => V.tip(e, `숫자 ${k} · 라운드 ${r.round}`, [['전역 모델 정확도', V.fmt.pct(a)], ['보유 클라이언트', owners[k]]])).on('mouseleave', V.untip)));
  d3.range(K).forEach(k => {
    f.g.append('text').attr('x', -6).attr('y', k * rh + rh / 2 + 3).attr('text-anchor', 'end').text('숫자 ' + k);
    f.g.append('text').attr('x', f.w + 4).attr('y', k * rh + rh / 2 + 3).text(`${V.fmt.pct(rs[rs.length - 1].class_acc[k])} · ${owners[k]}명`);
  });
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`)
    .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % Math.ceil(rs.length / 12) === 0)).tickSizeOuter(0));
  V.colorbar(f.g, cs, 0, f.h + 18, 100, '', V.fmt.pct);
}

const statusLegendItems = () => [{ type: 'ring', color: ENC.status.timeout, dash: true, name: '타임아웃' }, { type: 'ring', color: ENC.alert, name: '이상치' },
  { type: 'ring', color: 'var(--ink-2)', fill: 'var(--surface)', name: '미참여' }, { type: 'cross', name: '악성' }];

function renderTree(v, el, on) {
  el = V.clear(el);
  const root = d3.hierarchy(v.network.tree), fade = fadeOf(on), cl = S.snap.clients;
  V.legend(el, [{ type: 'sym', sym: d3.symbolStar, name: '서버' }, { type: 'sym', sym: d3.symbolSquare, name: '엣지' }].concat(statusLegendItems()));
  const showLeaf = (V.avail(el) - 8) / Math.max(1, root.leaves().length) >= 8;
  const f = V.frame(el, null, { l: 16, r: showLeaf ? 26 : 8, t: 6, b: 4 });
  d3.tree().size([f.h, f.w])(root);
  const leafGap = f.h / Math.max(1, root.leaves().length);
  f.g.selectAll('path').data(root.links()).join('path').attr('fill', 'none')
    .attr('stroke', d => d.target.data.status === 'timeout' ? ENC.status.timeout : 'var(--axis)')
    .attr('stroke-dasharray', d => d.target.data.status === 'timeout' ? '3 2' : null).attr('stroke-width', 1.1)
    .attr('opacity', d => d.target.depth === 2 ? fade(d.target.data.id) : (edgeOn(d.target.data.id) ? 1 : 0.15))
    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));
  const node = f.g.selectAll('g.n').data(root.descendants()).join('g').attr('transform', d => `translate(${d.y},${d.x})`).style('cursor', d => d.depth ? 'pointer' : null);
  node.each(function (d) {
    const g = d3.select(this);
    if (d.depth === 0) serverMark(g, 120);
    else if (d.depth === 1) { edgeMark(g, d.data.id, 80); g.attr('opacity', edgeOn(d.data.id) ? 1 : 0.2); }
    else { clientDot(g, d.data.id, Math.min(leafGap / 2.4, 3 + Math.sqrt(d.data.n || 1) / 20), statusOf(d.data), cl[d.data.id].malicious); g.attr('opacity', fade(d.data.id)); }
  });
  node.filter(d => d.depth < 2 || leafGap >= 8).append('text').attr('dx', d => d.children ? -8 : 7).attr('dy', 3)
    .attr('text-anchor', d => d.children ? 'end' : 'start').attr('font-size', d => d.depth === 2 ? Math.min(9, leafGap) : 10)
    .attr('opacity', d => d.depth === 2 ? fade(d.data.id) : 1)
    .text(d => d.depth === 1 ? `e${d.data.id} ${d.data.accepted}/${d.data.assigned}` : d.depth === 0 ? '서버' : 'c' + d.data.id);
  node.on('click', (e, d) => { if (d.depth === 1) select('edge', d.data.id); else if (d.depth === 2) select('client', d.data.id); })
    .on('mousemove', (e, d) => V.tip(e, d.depth === 2 ? `client ${d.data.id}` : d.data.name, d.depth === 2
      ? [['상태', STATUS_KO[statusOf(d.data)]], ['보유 숫자', (cl[d.data.id].classes || []).join(',')], ['지연', V.fmt.f1(d.data.latency || 0) + ' ms'], ['품질 Q', V.fmt.f3(d.data.quality || 0)], ['클릭', '모든 시각화에서 강조']]
      : d.depth === 1 ? [['상태', STATUS_KO[d.data.status] || d.data.status], ['서버 지연', V.fmt.f1(d.data.latency) + ' ms'], ['반영/배정', `${d.data.accepted}/${d.data.assigned}`], ['클릭', '이 엣지의 클라이언트 강조']] : [['엣지 수', d.children.length]]))
    .on('mouseleave', V.untip);
}

function renderHyper(v, el, on) {
  el = V.clear(el);
  const root = d3.hierarchy(v.network.tree), fade = fadeOf(on), cl = S.snap.clients;
  const cs = V.seq([0, d3.max(root.descendants().filter(d => d.depth > 0), d => d.data.latency || 0) || 1]);
  V.legend(el, [{ name: '링크 지연 (파랑 농도)', color: V.css('--seq-hi') }, { type: 'ln', color: ENC.status.timeout, dash: '3 2', name: '타임아웃' }].concat(statusLegendItems().slice(1, 2)));
  const H = V.avail(el), W = el.clientWidth || 400, RX = W / 2 - 10, RY = H / 2 - 10;  // 카드를 채우도록 타원으로 투영
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H);
  const g = svg.append('g').attr('transform', `translate(${W / 2},${H / 2})`);
  g.append('ellipse').attr('rx', RX).attr('ry', RY).attr('fill', 'var(--surface-2)').attr('stroke', 'var(--axis)');
  d3.cluster().size([2 * Math.PI, 1])(root);
  root.each(d => { const rr = Math.tanh(0.85 * d.depth / 2); d.z = [rr * Math.cos(d.x), rr * Math.sin(d.x)]; });
  const mobius = (z, a) => {
    const nr = [z[0] - a[0], z[1] - a[1]];
    const dr = [1 - (a[0] * z[0] + a[1] * z[1]), -(a[0] * z[1] - a[1] * z[0])];
    const den = dr[0] * dr[0] + dr[1] * dr[1];
    return [(nr[0] * dr[0] + nr[1] * dr[1]) / den, (nr[1] * dr[0] - nr[0] * dr[1]) / den];
  };
  const focusKey = S.sel ? (S.sel.kind + S.sel.id) : S.hyperFocus;
  const focus = root.descendants().find(d => (d.data.kind + (d.data.id ?? '')) === focusKey) || root;
  const P = d => { const w = mobius(d.z, focus.z); return [w[0] * RX, w[1] * RY, 1 - (w[0] * w[0] + w[1] * w[1])]; };
  g.selectAll('line').data(root.links()).join('line').each(function (l) {
    const a = P(l.source), b = P(l.target);
    d3.select(this).attr('x1', a[0]).attr('y1', a[1]).attr('x2', b[0]).attr('y2', b[1]);
  }).attr('stroke', l => l.target.data.status === 'timeout' ? ENC.status.timeout : cs(l.target.data.latency || 0))
    .attr('stroke-width', l => 0.6 + 2.2 * P(l.target)[2]).attr('stroke-dasharray', l => l.target.data.status === 'timeout' ? '3 2' : null)
    .attr('opacity', l => l.target.depth === 2 ? fade(l.target.data.id) : (edgeOn(l.target.data.id) ? 1 : 0.15));
  g.selectAll('g.n').data(root.descendants()).join('g').attr('transform', d => { const p = P(d); return `translate(${p[0]},${p[1]})`; })
    .each(function (d) {
      const s = Math.max(0.5, P(d)[2]), gg = d3.select(this).style('cursor', 'pointer');
      if (d.depth === 0) serverMark(gg, 110 * s);
      else if (d.depth === 1) { edgeMark(gg, d.data.id, 80 * s); gg.attr('opacity', edgeOn(d.data.id) ? 1 : 0.2); }
      else { clientDot(gg, d.data.id, 4.5 * s, statusOf(d.data), cl[d.data.id].malicious); gg.attr('opacity', fade(d.data.id)); }
      if (s > 0.6 || d.depth < 2) gg.append('text').attr('dy', -7 * s - 3).attr('text-anchor', 'middle').attr('font-size', 7 + 3 * s).text(d.depth === 0 ? '서버' : d.depth === 1 ? 'e' + d.data.id : 'c' + d.data.id);
    })
    .on('click', (e, d) => { if (d.depth === 2) select('client', d.data.id); else if (d.depth === 1) select('edge', d.data.id); else { S.hyperFocus = 'server'; S.sel = null; renderSelChip(); render(); } })
    .on('mousemove', (e, d) => V.tip(e, d.depth === 2 ? `client ${d.data.id}` : d.data.name, [['상태', STATUS_KO[statusOf(d.data)] || ''], ['지연', V.fmt.f1(d.data.latency || 0) + ' ms'], ['클릭', '선택 · 초점 이동']]))
    .on('mouseleave', V.untip);
}

function renderRank(v, el, on) {
  el = V.clear(el);
  const rk = v.network.ranks, n = S.snap.clients.length, fade = fadeOf(on);
  if (!rk.round.filter((r, k) => rk.rank[k]).length) return V.empty(el);
  const f = V.frame(el, null, { r: 26, l: 26, b: 20 });
  const x = d3.scaleLinear().domain(d3.extent(rk.round)).range([0, f.w]);
  const y = d3.scalePoint().domain(d3.range(1, n + 1)).range([0, f.h]);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(V.intTicks(x, 6)).tickSizeOuter(0));
  f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).tickValues(y.domain().filter(d => d === 1 || d % 5 === 0)));
  const lastK = rk.rank.length - 1;
  f.g.selectAll('path.rk').data(d3.range(n)).join('path').attr('class', 'rk').attr('fill', 'none')
    .attr('stroke', i => clientColor(i)).attr('stroke-width', i => on && on.has(i) ? 2.6 : 1.4).attr('opacity', i => on ? fade(i) : 0.75)
    .attr('d', i => d3.line().curve(d3.curveMonotoneX).x((r, k) => x(rk.round[k])).y((r, k) => y(rk.rank[k][i]))(rk.rank));
  f.g.selectAll('text.rl').data(d3.range(n)).join('text').attr('class', 'rl').attr('x', f.w + 4)
    .attr('y', i => y(rk.rank[lastK][i]) + 3).attr('font-size', 8.5).attr('fill', i => clientColor(i)).attr('font-weight', 600).text(i => 'c' + i)
    .attr('opacity', i => on ? (on.has(i) ? 1 : 0) : (rk.rank[lastK][i] <= 3 ? 1 : 0));
  const hit = i => [`client ${i}`, [['현재 순위', rk.rank[lastK][i]], ['품질 Q', V.fmt.f3(rk.quality[lastK][i])], ['소속', edgeLabel(lastAssign()[i])]]];
  f.g.append('rect').attr('width', f.w).attr('height', f.h).attr('fill', 'transparent').lower()
    .on('mousemove', e => {
      const [mx, my] = d3.pointer(e);
      const k = d3.leastIndex(rk.round, r => Math.abs(x(r) - mx));
      const r = Math.max(1, Math.min(n, Math.round(my / (f.h / (n - 1 || 1))) + 1));
      const i = rk.rank[k].indexOf(r);
      if (i >= 0) V.tip(e, `client ${i}`, [['라운드', rk.round[k]], ['우선순위', r], ['품질 Q', V.fmt.f3(rk.quality[k][i])], ['클릭', '선택']]);
      S._rankHover = i;
    })
    .on('click', () => { if (S._rankHover >= 0) select('client', S._rankHover); })
    .on('mouseleave', V.untip);
}

function renderMatchTrend(el) {
  const rs = S.snap.rounds; if (!rs.length) return V.empty(el);
  const ne = S.snap.topology.edges.length, x = rs.map(r => r.round);
  const per = j => rs.map(r => { const v = r.clients.filter(c => c.edge === j && c.quality != null).map(c => c.quality); return v.length ? d3.mean(v) : null; });
  const series = d3.range(ne).map(j => ({ name: `e${j} 평균 품질 Q`, values: per(j), color: ENC.ink, dash: edgeDash(j), width: edgeOn(j) && S.sel ? 2.6 : 1.8, opacity: edgeOn(j) ? 1 : 0.15 }));
  const ms = S.snap.views.network.matchings;
  V.line(el, { x, series, yDomain: [0, 1], yFmt: V.fmt.pct,
    legendItems: d3.range(ne).map(j => ({ type: 'ln', dash: edgeDash(j), name: `${edgeGlyph(j)}e${j}` })).concat([{ type: 'ln', color: 'var(--axis)', dash: '2 3', width: 1, name: '재매칭 (↺ 재배정 수)' }]),
    under: (f, xs) => ms.forEach((m, k) => {
      const moved = k > 0 ? m.assign.filter((a, i) => a !== ms[k - 1].assign[i]).length : null;
      f.g.append('line').attr('x1', xs(m.round)).attr('x2', xs(m.round)).attr('y1', 0).attr('y2', f.h).attr('stroke', 'var(--axis)').attr('stroke-dasharray', '2 3');
      f.g.append('text').attr('x', xs(m.round) + 2).attr('y', f.h - 3).attr('font-size', 8.5).text(moved == null ? '' : `↺${moved}${m.blocking ? '·b' + m.blocking : ''}`);
    }) });
}

function renderMatchOutliers(el, on) {
  el = V.clear(el);
  const rs = S.snap.rounds; if (!rs.length) return V.empty(el);
  const mal = new Set(S.snap.clients.filter(c => c.malicious).map(c => c.id));
  const pts = [];
  rs.forEach(r => {
    d3.groups(r.clients.filter(c => c.participated && c.quality != null && c.edge >= 0), c => c.edge).forEach(([j, cs]) => {
      if (cs.length < 3) return;
      const q = cs.map(c => c.quality), med = d3.median(q), mad = d3.median(q, v => Math.abs(v - med)) || 1e-3;
      cs.forEach(c => { const z = Math.max(0, 0.6745 * (med - c.quality) / mad); pts.push({ round: r.round, id: c.id, edge: j, z, q: c.quality, med, out: z > 3 }); });
    });
  });
  if (!pts.length) return V.empty(el, '엣지당 참여 클라이언트 3명 이상 필요');
  V.legend(el, [{ type: 'ring', color: ENC.alert, name: `이상치 z>3 (${pts.filter(p => p.out).length})` }, { type: 'cross', name: '악성 설정' }, { name: '점 색 = 클라이언트', color: 'var(--ink-2)' }]);
  scatterZ(el, pts, p => p.id, 3, '엣지 내 품질 저하 z', p => p.out, mal, on,
    p => [`client ${p.id} · ${edgeLabel(p.edge)} · R${p.round}`, [['품질 Q', V.fmt.f3(p.q)], ['엣지 중앙값', V.fmt.f3(p.med)], ['z', V.fmt.f2(p.z)], ['악성 설정', mal.has(p.id) ? '예' : '아니오']]]);
}

function scatterZ(el, pts, idOf, thr, ylab, isOut, mal, on, tipOf) {
  const f = V.frame(el, null, { t: 14, b: 20, l: 34 }), fade = fadeOf(on);
  const x = d3.scaleLinear().domain(d3.extent(pts, p => p.round)).range([6, f.w - 6]);
  const y = d3.scaleSymlog().domain([0, Math.max(5, d3.max(pts, p => p.z))]).range([f.h, 0]).nice();
  V.axes(f, x, y, { yLabel: ylab, yFmt: d3.format('~g'), yTicks: 4 });
  f.g.append('line').attr('x1', 0).attr('x2', f.w).attr('y1', y(thr)).attr('y2', y(thr)).attr('stroke', ENC.alert).attr('stroke-dasharray', '4 3');
  const jit = d3.scaleLinear().domain([0, S.snap.clients.length]).range([-4, 4]);
  f.g.selectAll('g.pt').data(pts).join('g').attr('transform', p => `translate(${x(p.round) + jit(idOf(p))},${y(p.z)})`).style('cursor', 'pointer').each(function (p) {
    const g = d3.select(this), i = idOf(p);
    g.append('circle').attr('r', isOut(p) ? 3.4 : 2.6).attr('fill', clientColor(i)).attr('stroke', 'var(--surface)').attr('stroke-width', 0.6);
    if (isOut(p)) g.append('circle').attr('r', 5.4).attr('fill', 'none').attr('stroke', ENC.alert).attr('stroke-width', 1.5);
    if (mal.has(i)) cross(g, 2.6);
    g.attr('opacity', on ? fade(i) : (isOut(p) ? 1 : 0.6));
  }).on('click', (e, p) => select('client', idOf(p))).on('mousemove', (e, p) => { const [t, rows] = tipOf(p); V.tip(e, t, rows); }).on('mouseleave', V.untip);
}

function clientOrder() {
  const a = lastAssign();
  return d3.range(S.snap.clients.length).sort((i, j) => ((a[i] ?? 99) - (a[j] ?? 99)) || i - j);
}

function renderFacets(el, on) {
  el = V.clear(el);
  const ms = S.snap.views.network.matchings;
  if (!ms.length) return V.empty(el, '재매칭 기록 없음');
  const key = S.view.facet, label = key === 'profit' ? '이익 π' : '엣지 선호 s', fade = fadeOf(on);
  const order = clientOrder(), n = order.length, ne = ms[0].profit[0].length;
  const cs = V.seq(d3.extent(ms.flatMap(m => m[key].flat())));
  V.legend(el, [{ type: 'sym', sym: d3.symbolCircle, name: '배정' }, { style: 'outline', color: ENC.ink, name: '직전 대비 재배정' }, { name: `파랑 농도 = ${label}`, color: V.css('--seq-hi') }]);
  const W = el.clientWidth || 600, H = V.avail(el), labelW = 26, head = 22, foot = 12, gapW = 8;
  let best = null;
  for (let r = 1; r <= Math.min(4, ms.length); r++) {
    const c = Math.ceil(ms.length / r);
    const cw = ((W - labelW) / c - gapW) / ne, chh = (H - r * (head + foot)) / (r * n);
    if (cw < 5 || chh < 1.5) continue;
    if (!best || cw * chh > best.cw * best.chh) best = { r, c, cw, chh };
  }
  best = best || { r: 1, c: ms.length, cw: Math.max(3, ((W - labelW) / ms.length - gapW) / ne), chh: Math.max(1.5, (H - head - foot) / n) };
  const cols = best.c, cellW = best.cw, cellH = best.chh;
  const facetW = ne * cellW + gapW, facetH = n * cellH + head + foot;
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H);
  ms.forEach((m, idx) => {
    const r = Math.floor(idx / cols), c = idx % cols, prev = idx > 0 ? ms[idx - 1].assign : null;
    const g = svg.append('g').attr('transform', `translate(${labelW + c * facetW},${r * facetH + head})`);
    const moved = prev ? m.assign.filter((a, i) => a !== prev[i]).length : 0;
    g.append('text').attr('x', 0).attr('y', -11).attr('font-size', 9.5).attr('fill', 'var(--ink)').attr('font-weight', 600).text(`R${m.round}`);
    g.append('text').attr('x', 0).attr('y', -2).attr('font-size', 8.5).attr('fill', 'var(--muted)').text(`${m.blocking ? 'blk' + m.blocking : '안정'}${prev ? ' ↺' + moved : ''}`);
    order.forEach((i, ri) => {
      const row = g.append('g').attr('opacity', fade(i)).style('cursor', 'pointer').on('click', () => select('client', i));
      for (let j = 0; j < ne; j++) {
        const assigned = m.assign[i] === j, changed = assigned && prev && prev[i] !== j;
        row.append('rect').attr('x', j * cellW).attr('y', ri * cellH).attr('width', cellW - 1).attr('height', Math.max(1, cellH - (cellH > 3 ? 0.5 : 0)))
          .attr('fill', cs(m[key][i][j])).attr('stroke', changed ? ENC.ink : null).attr('stroke-width', changed ? 1.4 : 0)
          .on('mousemove', e => V.tip(e, `R${m.round} · client ${i} → ${edgeLabel(j)}`, [[label, V.fmt.f2(m[key][i][j])],
            ['배정', assigned ? '예' + (changed ? ' (재배정)' : '') : '아니오'], ['엣지 내 우선순위', assigned ? m.edge_rank[i] : '–'], ['품질 Q', V.fmt.f3(m.Q[i])]]))
          .on('mouseleave', V.untip);
        if (assigned) row.append('circle').attr('cx', j * cellW + cellW / 2 - 0.5).attr('cy', ri * cellH + cellH / 2)
          .attr('r', Math.max(1, Math.min(4, cellH / 3, cellW / 3))).attr('fill', ENC.ink).attr('pointer-events', 'none');
      }
    });
    for (let j = 0; j < ne; j++) g.append('text').attr('x', j * cellW + cellW / 2).attr('y', n * cellH + 9).attr('text-anchor', 'middle').attr('font-size', 8).attr('opacity', edgeOn(j) ? 1 : 0.25).text(edgeGlyph(j) + j);
    if (c === 0) order.forEach((i, ri) => { if (cellH >= 7 || ri % Math.ceil(7 / cellH) === 0) g.append('text').attr('x', -3).attr('y', ri * cellH + cellH - 0.5).attr('text-anchor', 'end').attr('font-size', Math.min(8.5, cellH + 2)).attr('fill', clientColor(i)).attr('font-weight', 600).attr('opacity', fade(i)).text('c' + i); });
  });
}

function renderPixels(el, on) {
  el = V.clear(el);
  const rs = S.snap.rounds; if (!rs.length) return V.empty(el);
  const order = clientOrder(), ne = S.snap.topology.edges.length, R = rs.length, assign = lastAssign(), fade = fadeOf(on);
  const lat = rs.flatMap(r => r.clients.map(c => c.latency_ms).filter(x => x != null)).sort(d3.ascending);
  const byt = rs.flatMap(r => r.clients.map(c => c.bytes_up || 0));
  const csLat = V.seq([0, d3.quantile(lat, 0.98) || 1]), csB = V.seq([0, d3.max(byt) || 1]);
  const csE = V.seq(d3.extent(rs.flatMap(r => r.edges.map(e => e.latency_ms))));
  const ST = ENC.status;
  const panels = [
    { title: '통신 지연 (파랑 농도)', cell: c => c.status === 'idle' ? ST.idle : c.status === 'timeout' ? ST.timeout : csLat(c.latency_ms || 0), edge: e => csE(e.latency_ms), scale: csLat, unit: 'ms' },
    { title: '통신 상태', cell: c => c.outlier ? ST.outlier : (ST[c.status] || ST.idle), edge: e => e.status === 'idle' ? ST.idle : (e.status === 'down' ? ST.timeout : ST.ok) },
    { title: '업로드량 (파랑 농도)', cell: c => c.status === 'idle' ? ST.idle : csB(c.bytes_up || 0), edge: e => csB(e.bytes_up || 0), scale: csB, unit: 'KB' },
  ];
  V.legend(el, [{ color: ST.ok, name: '정상' }, { color: ST.timeout, name: '타임아웃' }, { color: ST.outlier, name: '이상치' }, { color: ST.idle, border: true, name: '미참여' }, { name: '행 라벨 색 = 클라이언트', color: 'var(--ink-2)' }]);
  const W = el.clientWidth || 800, H = V.avail(el), labelW = 34, gap = 12, sepH = 3, top = 13, bottom = 20;
  const pw = (W - labelW - gap * (panels.length - 1)) / panels.length, px = pw / R;
  const ph = Math.max(1.5, (H - top - bottom - (ne + 1) * sepH) / (order.length + ne));
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H);
  const rows = [];
  let y = top;
  d3.range(ne).forEach(j => { rows.push({ kind: 'edge', id: j, y }); y += ph; });
  y += sepH;
  d3.range(ne).forEach(j => {
    order.filter(i => assign[i] === j).forEach(i => { rows.push({ kind: 'client', id: i, y }); y += ph; });
    y += sepH;
  });
  order.filter(i => assign[i] == null || assign[i] < 0).forEach(i => { rows.push({ kind: 'client', id: i, y }); y += ph; });
  const xr = d3.scaleLinear().domain([rs[0].round, rs[R - 1].round]);
  const rowOp = row => row.kind === 'edge' ? (edgeOn(row.id) ? 1 : 0.15) : fade(row.id);
  panels.forEach((p, k) => {
    const g = svg.append('g').attr('transform', `translate(${labelW + k * (pw + gap)},0)`);
    g.append('rect').attr('width', pw).attr('height', y - top).attr('y', top).attr('fill', 'none').attr('stroke', 'var(--grid)');
    g.append('text').attr('x', 0).attr('y', 9).attr('font-size', 10).attr('fill', 'var(--ink)').text(p.title);
    rows.forEach(row => {
      const rg = g.append('g').attr('opacity', rowOp(row)).style('cursor', 'pointer').on('click', () => select(row.kind, row.id));
      rs.forEach((r, t) => {
        const d = row.kind === 'edge' ? r.edges[row.id] : r.clients[row.id];
        if (!d) return;
        rg.append('rect').attr('x', t * px).attr('y', row.y).attr('width', Math.max(1, px - (px > 4 ? 0.5 : 0))).attr('height', ph - (ph > 3 ? 0.4 : 0))
          .attr('fill', row.kind === 'edge' ? p.edge(d) : p.cell(d))
          .on('mousemove', e => V.tip(e, `${row.kind === 'edge' ? edgeLabel(row.id) : 'client ' + row.id} · 라운드 ${r.round}`, row.kind === 'edge'
            ? [['상태', STATUS_KO[d.status] || d.status], ['서버 지연', V.fmt.f1(d.latency_ms) + ' ms'], ['반영/배정', `${d.accepted}/${d.assigned}`], ['업로드', V.fmt.f1((d.bytes_up || 0) / 1024) + ' KB']]
            : [['상태', STATUS_KO[statusOf(d)] || d.status], ['소속', edgeLabel(d.edge)], ['지연', d.latency_ms == null ? '–' : V.fmt.f1(d.latency_ms) + ' ms'], ['업로드', V.fmt.f1((d.bytes_up || 0) / 1024) + ' KB']]))
          .on('mouseleave', V.untip);
      });
    });
    g.append('g').attr('class', 'axis').attr('transform', `translate(0,${y + 1})`)
      .call(d3.axisBottom(xr.copy().range([px / 2, pw - px / 2])).ticks(V.intTicks(xr, 5)).tickSizeOuter(0));
    if (p.scale) V.colorbar(g, p.scale, pw - Math.min(90, pw / 2), 1, Math.min(90, pw / 2), '', d => p.unit === 'KB' ? d3.format('.0f')(d / 1024) : d3.format('.0f')(d) + p.unit);
  });
  rows.forEach((row, k) => {
    if (row.kind === 'edge' || ph >= 6 || k % Math.ceil(6 / ph) === 0)
      svg.append('text').attr('x', labelW - 3).attr('y', row.y + ph - 0.5).attr('text-anchor', 'end').attr('font-size', Math.min(8.5, ph + 2))
        .attr('fill', row.kind === 'edge' ? ENC.ink : clientColor(row.id)).attr('font-weight', 600).attr('opacity', rowOp(row))
        .text(row.kind === 'edge' ? edgeGlyph(row.id) + row.id : 'c' + row.id);
  });
}

function renderParallel(el, on) {
  el = V.clear(el);
  const rs = S.snap.rounds; if (!rs.length) return V.empty(el);
  const cl = S.snap.clients, assign = lastAssign();
  const mean = a => { const b = a.filter(x => x != null); return b.length ? d3.mean(b) : null; };
  const data = cl.map((c, i) => {
    const h = rs.map(r => r.clients[i]);
    const part = h.filter(x => x.participated);
    return { id: i, edge: assign[i], malicious: c.malicious,
      bandwidth: c.bandwidth_mbps, lat_mean: mean(part.map(x => x.latency_ms)), lat_max: d3.max(part, x => x.latency_ms) ?? null,
      timeouts: h.filter(x => x.status === 'timeout').length, participation: part.length / rs.length,
      upload: d3.sum(h, x => x.bytes_up || 0) / 2 ** 20, z: mean(part.map(x => x.z)),
      outliers: h.filter(x => x.outlier).length, quality: h[h.length - 1].quality };
  });
  const dims = [['bandwidth', '대역폭 Mbps'], ['lat_mean', '평균 지연 ms'], ['lat_max', '최대 지연 ms'], ['timeouts', '타임아웃'],
    ['participation', '참여율'], ['upload', '업로드 MB'], ['z', '업데이트 z'], ['outliers', '이상치'], ['quality', '품질 Q']];
  V.legend(el, [{ name: '선 색 = 클라이언트', color: 'var(--ink-2)', type: 'ln' }, { type: 'cross', name: '악성 설정 (시작점)' }, { name: '축 드래그 = 범위 선택', color: 'var(--axis)' }]);
  const f = V.frame(el, null, { t: 16, b: 6, l: 26, r: 26 });
  const x = d3.scalePoint().domain(dims.map(d => d[0])).range([0, f.w]);
  const y = {};
  dims.forEach(([k]) => { const ext = d3.extent(data, d => d[k]); y[k] = d3.scaleLinear().domain(ext[0] == null ? [0, 1] : ext[0] === ext[1] ? [ext[0] - 1, ext[1] + 1] : ext).nice().range([f.h, 0]); });
  const line = d => d3.line().defined(p => p[1] != null)(dims.map(([k]) => [x(k), d[k] == null ? null : y[k](d[k])]));
  const base = d => on ? (on.has(d.id) ? 1 : 0.08) : 0.8;
  const paths = f.g.append('g').selectAll('path').data(data).join('path').attr('d', line).attr('fill', 'none')
    .attr('stroke', d => clientColor(d.id)).attr('stroke-width', d => on && on.has(d.id) ? 2.6 : 1.5).attr('opacity', base).style('cursor', 'pointer')
    .on('click', (e, d) => select('client', d.id))
    .on('mousemove', (e, d) => V.tip(e, `client ${d.id} · ${edgeLabel(d.edge)}`, dims.map(([k, lbl]) => [lbl, d[k] == null ? '–' : V.fmt.auto(d[k])])))
    .on('mouseleave', V.untip);
  data.filter(d => d.malicious && d.bandwidth != null).forEach(d => cross(f.g.append('g').attr('transform', `translate(${x('bandwidth')},${y.bandwidth(d.bandwidth)})`).attr('opacity', base(d)), 3.5));
  const brushes = {};
  const update = () => {
    const act = Object.entries(brushes).filter(([, r]) => r);
    paths.attr('opacity', d => act.length ? (act.every(([k, [a, b]]) => d[k] != null && y[k](d[k]) >= a && y[k](d[k]) <= b) ? 0.9 : 0.05) : base(d));
  };
  dims.forEach(([k, lbl]) => {
    const g = f.g.append('g').attr('transform', `translate(${x(k)},0)`);
    g.append('g').attr('class', 'axis').call(d3.axisLeft(y[k]).ticks(4).tickSizeOuter(0));
    g.append('text').attr('y', -5).attr('text-anchor', 'middle').attr('font-size', 9.5).attr('fill', 'var(--ink)').text(lbl);
    g.append('g').call(d3.brushY().extent([[-9, 0], [9, f.h]]).on('brush end', e => { brushes[k] = e.selection; update(); }));
  });
}

function groupByAlg(runs) {
  const algs = [...new Set(runs.map(r => r.algorithm))].sort((a, b) => (ALG_ORDER[a] ?? 9) - (ALG_ORDER[b] ?? 9));
  return algs.map(a => ({ alg: a, name: ALG_NAME[a] || a, ...(ENC.alg[a] || { dash: null, marker: d3.symbolCircle, bar: null }), runs: runs.filter(r => r.algorithm === a) }));
}
const mstd = vals => { const v = vals.filter(x => x != null && isFinite(x)); return v.length ? { m: d3.mean(v), s: v.length > 1 ? d3.deviation(v) : 0, n: v.length } : null; };
const bandPath = (f, xs, ys, x, b) => f.g.append('path').datum(b).attr('fill', ENC.ink).attr('opacity', 0.07)
  .attr('d', d3.area().defined(q => q[1] != null).x((q, k) => xs(x[k])).y0(q => ys(q[1])).y1(q => ys(q[2])));

function algoDelta(el, A, g) {
  const ref = A.find(a => a.alg === 'fedavg');
  if (!ref) return V.empty(el, 'FedAvg 실행 없음');
  const seeds = [...new Set(g.runs.map(r => r.seed))];
  const T = d3.max(g.runs, r => r.round.length), x = d3.range(1, T + 1);
  const at = (r, k) => r.accuracy[k] ?? null;
  const D = A.filter(a => a !== ref).map(a => ({ ...a, d: x.map((_, k) => {
    const v = seeds.map(s => { const r1 = a.runs.find(r => r.seed === s), r0 = ref.runs.find(r => r.seed === s);
      return r1 && r0 && at(r1, k) != null && at(r0, k) != null ? at(r1, k) - at(r0, k) : null; }).filter(q => q != null);
    return v.length ? [d3.mean(v), d3.min(v), d3.max(v)] : [null, null, null]; }) }));
  const ext = d3.max(D.flatMap(a => a.d.flatMap(q => [Math.abs(q[1] ?? 0), Math.abs(q[2] ?? 0)]))) || 0.05;
  V.line(el, { x, yDomain: [-ext * 1.1, ext * 1.1], yFmt: d3.format('+.1%'),
    series: D.map(a => ({ name: `${a.name} − FedAvg`, values: a.d.map(q => q[0]), color: ENC.ink, dash: a.dash, marker: a.marker })),
    under: (f, xs, ys) => {
      f.g.append('line').attr('x1', 0).attr('x2', f.w).attr('y1', ys(0)).attr('y2', ys(0)).attr('stroke', 'var(--ink-2)');
      f.g.append('text').attr('x', 4).attr('y', ys(0) - 3).attr('font-size', 9).text('= FedAvg');
      D.forEach(a => bandPath(f, xs, ys, x, a.d));
    } });
}

function algoCurves(el, A, g, curV) {
  const T = d3.max(g.runs, r => r.round.length), x = d3.range(1, T + 1);
  const B = A.map(a => ({ ...a, b: x.map((_, k) => { const v = a.runs.map(r => r.accuracy[k]).filter(q => q != null); return v.length ? [d3.mean(v), d3.min(v), d3.max(v)] : [null, null, null]; }) }));
  const series = B.map(a => ({ name: a.name, values: a.b.map(q => q[0]), color: ENC.ink, dash: a.dash, marker: a.marker }));
  if (curV) series.push({ name: '이 실행', values: x.map((_, k) => curV.trend.accuracy[k] ?? null), color: 'var(--muted)', width: 3, opacity: 0.6 });
  V.line(el, { x, yDomain: [0, 1], yFmt: V.fmt.pct, series, under: (f, xs, ys) => B.forEach(a => bandPath(f, xs, ys, x, a.b)) });
}

function algoDigitBars(el, A) {
  const K = (A.flatMap(a => a.runs).find(r => r.class_acc) || {}).class_acc?.length || 0;
  if (!K) return V.empty(el, '숫자별 정확도 없음');
  V.bars(el, { groups: d3.range(K).map(String), keys: A.map(a => a.name), colors: A.map(() => ENC.ink), styles: A.map(a => a.bar), yDomain: [0, 1], yFmt: d3.format('.2f'), labels: false,
    values: d3.range(K).map(k => A.map(a => mstd(a.runs.map(r => r.class_acc && r.class_acc[k]))?.m ?? null)) });
}

function renderLdflCard(el) {
  const j = S.job;
  el.classList.remove('scroll');
  if (!j) return V.empty(el, '⚙ 설정 · 실행 → 지역분산 vs 연합 비교 실행');
  if (j.error) { V.clear(el).innerHTML = `<pre class="err">${j.error}</pre>`; return; }
  const envs = j.config.envs, label = k => j.labels[k] || k, short = k => ENV_SHORT[k] || label(k);
  const metricNames = { f1: 'F1', precision: '정밀도', recall: '재현율', accuracy: '정확도' };
  const chosen = $('ldflMetric').value, metric = metricNames[chosen] ? chosen : 'f1', metricName = metricNames[metric];
  if (S.view.ldfl === 'prog' || !j.report) {
    const ep = d3.max(envs, k => (j.progress[label(k)] || []).length) || 0;
    if (!ep) return V.empty(el, '대기 중…');
    const x = d3.range(1, (j.config.epochs || ep) + 1);
    return V.line(el, { x, xName: 'epoch', yDomain: [0, 1], yFmt: d3.format('.3f'),
      series: envs.map((k, i) => ({ name: short(k), color: ENC.ink, dash: ENV_DASH[i % ENV_DASH.length], values: x.map((_, q) => ((j.progress[label(k)] || [])[q] || {})[metric] ?? null) })) });
  }
  const res = j.report.results;
  if (S.view.ldfl === 'f1')
    return V.bars(el, { groups: res.map(r => short(r.key).replace('연합 ', '').replace('지역·', '지역 ')), keys: [`글로벌 ${metricName}`, `로컬 ${metricName}`],
      colors: [ENC.ink, ENC.ink], styles: [null, 'outline'], yDomain: [0, 1], yFmt: d3.format('.2f'), labels: false, values: res.map(r => [r.global[metric], r.local_mean[metric]]) });
  el.classList.add('scroll');
  V.clear(el).innerHTML = `<table class="tbl"><thead><tr><th>환경</th><th class="num">원 데이터 이동</th><th class="num">통신</th><th>원데이터 외부</th><th class="num">DP ε</th></tr></thead><tbody>${res.map(r => `<tr>
    <td title="${r.name}">${short(r.key)}</td>
    <td class="num">${V.fmt.mb(r.cost.raw_bytes / 2 ** 20)}</td><td class="num">${V.fmt.mb((r.cost.model_bytes + r.cost.stat_bytes) / 2 ** 20)}</td>
    <td>${r.privacy.raw_data_leaves_site ? badge('warn', '예') : badge('finished', '아니오')}</td><td class="num">${r.privacy.dp ? V.fmt.f1(r.privacy.dp.epsilon) : '–'}</td></tr>`).join('')}</tbody></table>`;
}

function renderGtrend(v, el) {
  const t = v.trend, rs = S.snap.rounds;
  const nOut = rs.map(r => r.clients.filter(c => c.outlier).length), nTo = rs.map(r => r.clients.filter(c => c.status === 'timeout').length);
  const mx = Math.max(1, d3.max(nOut), d3.max(nTo));
  V.line(el, { x: t.round, yDomain: [0, 1], yFmt: V.fmt.pct,
    legendItems: [{ type: 'ln', name: '정확도' }, { type: 'ln', dash: '6 3', width: 1.3, name: 'EMA' }, { type: 'ln', dash: '1.5 2.5', width: 1.3, name: '하위10%' },
      { color: ENC.alert, name: '이상치 수' }, { color: ENC.status.timeout, name: '타임아웃 수' }],
    series: [{ name: '정확도', values: t.accuracy, color: ENC.ink }, { name: '정확도 EMA', values: t.acc_ema, color: ENC.ink, dash: '6 3', width: 1.3 },
      { name: '하위 10% 클라이언트', values: t.fair_worst, color: ENC.ink, dash: '1.5 2.5', width: 1.3 }],
    markers: t.alerts.filter(a => a.type !== 'diverge').map(a => ({ x: a.round, y: t.accuracy[t.round.indexOf(a.round)], label: a.msg, color: ENC.alert })),
    under: (f, xs) => {
      const bw = Math.max(2, f.w / t.round.length * 0.35), bh = f.h * 0.22;
      t.round.forEach((r, k) => {
        if (nOut[k]) f.g.append('rect').attr('x', xs(r) - bw).attr('y', f.h - bh * nOut[k] / mx).attr('width', bw).attr('height', bh * nOut[k] / mx).attr('fill', ENC.alert);
        if (nTo[k]) f.g.append('rect').attr('x', xs(r)).attr('y', f.h - bh * nTo[k] / mx).attr('width', bw).attr('height', bh * nTo[k] / mx).attr('fill', ENC.status.timeout);
      });
    } });
}

function clientClusters(sim) {
  const root = d3.hierarchy(sim.tree);
  let thr = 0.5, groups;
  for (let it = 0; it < 6; it++) {
    groups = [];
    const walk = nd => { if (nd.data.height <= thr || !nd.children) groups.push(nd.leaves().map(l => l.data.client)); else nd.children.forEach(walk); };
    walk(root);
    if (groups.length <= 8) break;
    thr += 0.15;
  }
  groups.sort((a, b) => b.length - a.length || d3.min(a) - d3.min(b));
  const cl = {};
  groups.forEach((g, k) => g.forEach(c => { cl[c] = k; }));
  return { groups, cl };
}

function mds2(S2) {
  const n = S2.length, D2 = S2.map(r => r.map(s => (1 - s) ** 2));
  const rm = D2.map(r => d3.mean(r)), gm = d3.mean(rm);
  const B = D2.map((r, i) => r.map((d, j) => -0.5 * (d - rm[i] - rm[j] + gm)));
  const vecs = [];
  for (let k = 0; k < 2; k++) {
    let v = d3.range(n).map(i => Math.sin(i + 1 + k));
    let lam = 0;
    for (let it = 0; it < 100; it++) {
      let w = B.map(r => d3.sum(r, (b, j) => b * v[j]));
      vecs.forEach(([u, l]) => { const dot = d3.sum(u, (x, i) => x * v[i]); w = w.map((x, i) => x - l * dot * u[i]); });
      lam = Math.sqrt(d3.sum(w, x => x * x)) || 1e-9;
      v = w.map(x => x / lam);
    }
    vecs.push([v, lam]);
  }
  return d3.range(n).map(i => [vecs[0][0][i] * Math.sqrt(vecs[0][1]), vecs[1][0][i] * Math.sqrt(vecs[1][1])]);
}

function renderClientCmp(v, el, on) {
  el = V.clear(el);
  const sim = v.similarity;
  if (!sim || !sim.tree) return V.empty(el, '업데이트 스케치 부족 (2라운드 이후)');
  const { groups, cl } = clientClusters(sim), fade = fadeOf(on), mal = new Set(S.snap.clients.filter(c => c.malicious).map(c => c.id));
  const metric = $('cmpMetric').value, ser = v.clients.series[metric], rounds = S.snap.rounds.map(r => r.round);
  V.legend(el, [{ name: '점 색 = 클라이언트', color: 'var(--ink-2)' }, { type: 'ring', color: ENC.ink, dash: true, name: '군집 경계 (번호)' }, { name: '파랑 농도 = 학습 지표', color: V.css('--seq-hi') }]);
  const W = el.clientWidth || 600, H = V.avail(el), lw = Math.round(W * 0.38), gap = 14;
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H);
  const P = mds2(sim.matrix), ids = sim.ids, pad = 12;
  const q = (k, a) => d3.quantile(P.map(p => p[k]).sort(d3.ascending), a);
  const dom = k => { const lo = q(k, 0.05), hi = q(k, 0.95); return lo === hi ? [lo - 1, hi + 1] : [lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08]; };
  const xs = d3.scaleLinear().domain(dom(0)).range([pad, lw - pad]).clamp(true);
  const ys = d3.scaleLinear().domain(dom(1)).range([H - pad, pad]).clamp(true);
  svg.append('rect').attr('width', lw).attr('height', H).attr('fill', 'var(--surface-2)').attr('rx', 4);
  const pos = new Map(ids.map((c, i) => [c, [xs(P[i][0]), ys(P[i][1])]]));
  groups.forEach((gr, k) => {
    const pts = gr.map(c => pos.get(c));
    const hull = pts.length >= 3 ? d3.polygonHull(pts) : null;
    const cx = d3.mean(pts, p => p[0]), cy = d3.mean(pts, p => p[1]);
    if (hull) svg.append('path').attr('d', 'M' + hull.map(p => [cx + (p[0] - cx) * 1.25 + Math.sign(p[0] - cx) * 4, cy + (p[1] - cy) * 1.25 + Math.sign(p[1] - cy) * 4]).join('L') + 'Z')
      .attr('fill', 'none').attr('stroke', ENC.ink).attr('stroke-dasharray', '3 2').attr('stroke-width', 0.9).attr('opacity', 0.6);
    svg.append('text').attr('x', cx).attr('y', d3.min(pts, p => p[1]) - 8).attr('text-anchor', 'middle').attr('font-size', 9).attr('font-weight', 700).attr('fill', ENC.ink).text(`군집${k + 1}`);
  });
  const last = ser.map(s => [...s].reverse().find(qq => qq != null));
  const node = svg.append('g').selectAll('g').data(ids).join('g').attr('transform', c => `translate(${pos.get(c)[0]},${pos.get(c)[1]})`).style('cursor', 'pointer')
    .attr('opacity', c => fade(c));
  node.each(function (c) { clientDot(d3.select(this), c, 5, 'ok', mal.has(c)); });
  node.append('text').attr('x', 7).attr('y', 3).attr('font-size', 8.5).attr('fill', c => clientColor(c)).attr('font-weight', 600).text(c => 'c' + c);
  node.on('click', (e, c) => select('client', c))
    .on('mousemove', (e, c) => { const cs = S.snap.clients[c]; V.tip(e, `client ${c} · 군집 ${cl[c] + 1}`, [['보유 숫자', (cs.classes || []).join(', ')], ['소속', edgeLabel(lastAssign()[c])], [metric, last[c] == null ? '–' : V.fmt.auto(last[c])]]); })
    .on('mouseleave', V.untip);
  const order = groups.flat(), x0 = lw + gap + 46, mw = W - x0 - 2;
  const rh = (H - 16 - (groups.length - 1) * 3) / order.length, cw = mw / rounds.length;
  const all = ser.flat().filter(qq => qq != null);
  const isAcc = metric.endsWith('acc') || metric === 'quality';
  const cs = V.seq(isAcc ? [0, 1] : d3.extent(all));
  let y = 0;
  groups.forEach((gr, k) => {
    svg.append('text').attr('x', lw + gap).attr('y', y + Math.min(rh * gr.length, 10)).attr('font-size', 8.5).attr('font-weight', 700).attr('fill', ENC.ink).text(`군집${k + 1}`);
    gr.forEach(c => {
      const row = svg.append('g').attr('opacity', fade(c)).style('cursor', 'pointer').on('click', () => select('client', c));
      row.append('text').attr('x', x0 - 3).attr('y', y + rh / 2 + 3).attr('text-anchor', 'end').attr('font-size', Math.min(8.5, rh + 2)).attr('fill', clientColor(c)).attr('font-weight', 600).text(rh >= 6 ? 'c' + c : '');
      ser[c].forEach((val, t) => {
        if (val == null) return;
        row.append('rect').attr('x', x0 + t * cw).attr('y', y).attr('width', Math.max(1, cw - 0.4)).attr('height', Math.max(1, rh - 0.4)).attr('fill', cs(val))
          .on('mousemove', e => V.tip(e, `client ${c} · 군집 ${k + 1} · 라운드 ${rounds[t]}`, [[metric, V.fmt.auto(val)]])).on('mouseleave', V.untip);
      });
      y += rh;
    });
    y += 3;
  });
  svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - 14})`)
    .call(d3.axisBottom(d3.scaleLinear().domain(d3.extent(rounds)).range([x0 + cw / 2, x0 + mw - cw / 2])).ticks(V.intTicks(d3.scaleLinear().domain(d3.extent(rounds)), 6)).tickSizeOuter(0));
}

function renderHeat(d, el, on) {
  el = V.clear(el);
  const n = d.ratio.length, k = d.ratio[0].length, fade = fadeOf(on);
  V.legend(el, [{ name: '파랑 농도 = 숫자 구성비', color: V.css('--seq-hi') }, { name: '막대 색 = 클라이언트 (위: 비율 p, 아래: 엔트로피 H)', color: 'var(--ink-2)' }]);
  const f = V.frame(el, null, { l: 16, t: 2, b: 12, r: 2 });
  const bh = Math.max(8, f.h * 0.14), ch = (f.h - 2 * bh - 8) / k, cw = f.w / n;
  const cs = V.seq([0, d3.max(d.ratio.flat())]);
  const y0 = k * ch + 4, rmax = d3.max(d.data_ratio), emax = Math.log2(k);
  for (let i = 0; i < n; i++) {
    const col = f.g.append('g').attr('opacity', fade(i)).style('cursor', 'pointer').on('click', () => select('client', i));
    for (let c = 0; c < k; c++)
      col.append('rect').attr('x', i * cw).attr('y', c * ch).attr('width', Math.max(1, cw - 1)).attr('height', Math.max(1, ch - 1)).attr('fill', cs(d.ratio[i][c]))
        .on('mousemove', e => V.tip(e, `client ${i} · 숫자 ${c}`, [['구성비', V.fmt.pct(d.ratio[i][c])], ['표본 수', d.counts[i][c]]])).on('mouseleave', V.untip);
    const tipf = e => V.tip(e, `client ${i}`, [['데이터 수 n', d.n[i]], ['데이터 비율 p', V.fmt.pct(d.data_ratio[i])], ['엔트로피 H', V.fmt.f3(d.entropy[i]) + ' bit']]);
    col.append('rect').attr('x', i * cw).attr('y', y0 + bh - bh * d.data_ratio[i] / rmax).attr('width', Math.max(1, cw - 1)).attr('height', bh * d.data_ratio[i] / rmax).attr('fill', clientColor(i)).on('mousemove', tipf).on('mouseleave', V.untip);
    col.append('rect').attr('x', i * cw).attr('y', y0 + 2 * bh + 4 - bh * d.entropy[i] / emax).attr('width', Math.max(1, cw - 1)).attr('height', bh * d.entropy[i] / emax).attr('fill', clientColor(i)).on('mousemove', tipf).on('mouseleave', V.untip);
    if (cw > 12 || i % 5 === 0) col.append('text').attr('x', i * cw + cw / 2).attr('y', f.h + 11).attr('text-anchor', 'middle').attr('font-size', 8.5).attr('fill', clientColor(i)).attr('font-weight', 600).text('c' + i);
  }
  for (let c = 0; c < k; c++) if (ch >= 8 || c % 2 === 0) f.g.append('text').attr('x', -4).attr('y', c * ch + ch / 2 + 3).attr('text-anchor', 'end').attr('font-size', 9).text(c);
  f.g.append('text').attr('x', -4).attr('y', y0 + bh - 1).attr('text-anchor', 'end').attr('font-size', 9).text('p');
  f.g.append('text').attr('x', -4).attr('y', y0 + 2 * bh + 3).attr('text-anchor', 'end').attr('font-size', 9).text('H');
}

function renderDendro(sim, el, on) {
  el = V.clear(el);
  const order = sim.order, n = order.length, idx = new Map(sim.ids.map((c, i) => [c, i])), fade = fadeOf(on);
  const W = el.clientWidth || 400, H = V.avail(el), dw = Math.min(140, W * 0.28), mx = dw + 22;
  const cellW = (W - mx - 6) / n, cell = (H - 6) / n;  // 행 높이·열 너비를 각각 카드에 맞춤
  const f = V.frame(el, null, { l: 2, t: 2, r: 2, b: 2 });
  const root = d3.hierarchy(sim.tree);
  const xh = d3.scaleLinear().domain([root.data.height || 1, 0]).range([0, dw]);
  const pos = new Map(order.map((c, r) => [c, r * cell + cell / 2]));
  root.eachAfter(d => { d.py = d.children ? d3.mean(d.children, c => c.py) : pos.get(d.data.client); d.px = xh(d.data.height); });
  root.each(d => { if (d.children) d.children.forEach(c => f.g.append('path').attr('fill', 'none').attr('stroke', 'var(--ink-2)').attr('stroke-width', 0.9).attr('d', `M${d.px},${d.py}V${c.py}H${c.px}`)); });
  order.forEach((c, r) => { if (cell >= 7 || r % 2 === 0) f.g.append('text').attr('x', dw + 19).attr('y', r * cell + cell / 2 + 3).attr('text-anchor', 'end').attr('font-size', Math.min(8.5, cell + 2)).attr('fill', clientColor(c)).attr('font-weight', 600).attr('opacity', fade(c)).style('cursor', 'pointer').on('click', () => select('client', c)).text('c' + c); });
  const cs = V.div([-1, 1]);
  order.forEach((a, r) => order.forEach((b, qq) => {
    const s = sim.matrix[idx.get(a)][idx.get(b)];
    f.g.append('rect').attr('x', mx + qq * cellW).attr('y', r * cell).attr('width', Math.max(1, cellW - 0.5)).attr('height', Math.max(1, cell - 0.5)).attr('fill', cs(s))
      .attr('opacity', on ? (on.has(a) || on.has(b) ? 1 : 0.2) : 1)
      .on('mousemove', e => V.tip(e, `client ${a} · client ${b}`, [['코사인 유사도 (파랑 농도)', V.fmt.f3(s)]])).on('mouseleave', V.untip);
  }));
}

function renderSocio(sim, el, on) {
  el = V.clear(el);
  const cl = S.snap.clients, fade = fadeOf(on);
  V.legend(el, [{ name: '점 색 = 클라이언트', color: 'var(--ink-2)' }, { type: 'cross', name: '악성 설정' }, { type: 'ln', color: 'var(--axis)', name: '유사도 ≥ 0.5' }]);
  const W = el.clientWidth || 400, H = V.avail(el);
  const svg = d3.select(el).append('svg').attr('width', W).attr('height', H);
  const nodes = sim.ids.map(i => Object.assign({ id: i }, S.socioPos[i] || {}));
  const links = sim.links.map(l => ({ source: l.source, target: l.target, value: l.value }));
  const fresh = !nodes.every(nd => nd.x != null);
  const r0 = Math.max(4, Math.min(9, Math.sqrt(W * H / nodes.length) / 6));
  const simu = d3.forceSimulation(nodes).force('link', d3.forceLink(links).id(d => d.id).distance(l => 20 + 60 * (1 - l.value)).strength(l => l.value * 0.6))
    .force('charge', d3.forceManyBody().strength(-120)).force('center', d3.forceCenter(W / 2, H / 2)).force('collide', d3.forceCollide(r0 + 3))
    .force('x', d3.forceX(W / 2).strength(0.05)).force('y', d3.forceY(H / 2).strength(0.08)).stop();
  for (let i = 0; i < (fresh ? 300 : 40); i++) simu.tick();
  nodes.forEach(nd => { S.socioPos[nd.id] = { x: nd.x, y: nd.y }; });
  const ex = d3.extent(nodes, d => d.x), ey = d3.extent(nodes, d => d.y), pad = r0 + 3;
  const sx = d3.scaleLinear().domain(ex[0] === ex[1] ? [ex[0] - 1, ex[1] + 1] : ex).range([pad, W - pad]);
  const sy = d3.scaleLinear().domain(ey[0] === ey[1] ? [ey[0] - 1, ey[1] + 1] : ey).range([pad, H - pad]);
  nodes.forEach(nd => { nd.x = sx(nd.x); nd.y = sy(nd.y); });
  svg.append('g').selectAll('line').data(links).join('line').attr('x1', l => l.source.x).attr('y1', l => l.source.y).attr('x2', l => l.target.x).attr('y2', l => l.target.y)
    .attr('stroke', 'var(--axis)').attr('stroke-width', l => 0.5 + 2 * (l.value - 0.5) * 2).attr('opacity', l => on ? (on.has(l.source.id) || on.has(l.target.id) ? 1 : 0.1) : 1);
  const g = svg.append('g').selectAll('g').data(nodes).join('g').attr('transform', d => `translate(${d.x},${d.y})`).attr('opacity', d => fade(d.id)).style('cursor', 'pointer');
  g.each(function (d) { clientDot(d3.select(this), d.id, r0, 'ok', cl[d.id].malicious); });
  g.append('text').attr('dx', r0 + 2).attr('dy', 3).attr('font-size', 8).attr('fill', d => clientColor(d.id)).attr('font-weight', 600).attr('pointer-events', 'none').text(d => 'c' + d.id);
  g.on('click', (e, d) => select('client', d.id))
    .on('mousemove', (e, d) => V.tip(e, `client ${d.id}`, [['소속', edgeLabel(lastAssign()[d.id])], ['데이터 수', cl[d.id].n], ['연결 수', links.filter(l => l.source.id === d.id || l.target.id === d.id).length], ['악성 설정', cl[d.id].malicious ? '예' : '아니오']]))
    .on('mouseleave', V.untip);
}

function renderStream(v, el, on) {
  el = V.clear(el);
  const st = v.contribution.stream, n = S.snap.clients.length, fade = fadeOf(on);
  if (!st.round.length) return V.empty(el);
  const keys = d3.range(n);
  V.legend(el, [{ name: '층 색 = 클라이언트 (두께 = 기여 가중치)', color: 'var(--ink-2)' }]);
  const f = V.frame(el, null, { b: 18, l: 8, r: 8 });
  const data = st.round.map((r, k) => { const o = { round: r }; keys.forEach(i => o[i] = st.weights[k][i]); return o; });
  const series = d3.stack().keys(keys).offset(d3.stackOffsetSilhouette)(data);
  const x = d3.scaleLinear().domain(d3.extent(st.round)).range([0, f.w]);
  const y = d3.scaleLinear().domain([d3.min(series, s => d3.min(s, d => d[0])), d3.max(series, s => d3.max(s, d => d[1]))]).range([f.h, 0]);
  f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).ticks(V.intTicks(x, 6)).tickSizeOuter(0));
  const area = d3.area().curve(d3.curveMonotoneX).x(d => x(d.data.round)).y0(d => y(d[0])).y1(d => y(d[1]));
  f.g.selectAll('path.layer').data(series).join('path').attr('class', 'layer').attr('d', area)
    .attr('fill', s => clientColor(+s.key)).attr('opacity', s => on ? fade(+s.key) : 0.9).attr('stroke', 'var(--surface)').attr('stroke-width', 0.8).style('cursor', 'pointer')
    .on('click', (e, s) => select('client', +s.key))
    .on('mousemove', (e, s) => {
      const k = d3.leastIndex(st.round, r => Math.abs(x(r) - d3.pointer(e)[0]));
      V.tip(e, `client ${s.key}`, [['라운드', st.round[k]], ['기여 가중치', V.fmt.pct(st.weights[k][s.key])], ['소속', edgeLabel(lastAssign()[+s.key])]]);
    }).on('mouseleave', V.untip);
}

function renderPareto(v, el, on) {
  el = V.clear(el);
  const p = v.contribution.pareto, fade = fadeOf(on);
  if (!p.client.length) return V.empty(el);
  V.legend(el, [{ name: '막대 색 = 클라이언트 (기여 비율)', color: 'var(--ink-2)' }, { type: 'ln', name: '누적 비율' }]);
  const f = V.frame(el, null, { b: 18, l: 36 });
  const x = d3.scaleBand().domain(p.client.map(String)).range([0, f.w]).padding(0.15);
  const y = d3.scaleLinear().domain([0, 1]).range([f.h, 0]);
  f.g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(4).tickSize(-f.w).tickFormat(''));
  const ax = f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x).tickSizeOuter(0).tickFormat(c => 'c' + c));
  ax.selectAll('text').attr('font-size', 7.5).style('fill', c => clientColor(+c)).attr('font-weight', 600);
  f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickFormat(V.fmt.pct));
  f.g.append('line').attr('x1', 0).attr('x2', f.w).attr('y1', y(0.8)).attr('y2', y(0.8)).attr('stroke', 'var(--axis)').attr('stroke-dasharray', '4 3');
  f.g.selectAll('rect.b').data(p.client).join('rect').attr('class', 'b').attr('x', c => x(String(c))).attr('width', x.bandwidth())
    .attr('y', (c, i) => y(p.share[i])).attr('height', (c, i) => f.h - y(p.share[i])).attr('rx', 1.5).attr('fill', c => clientColor(c)).attr('opacity', c => fade(c)).style('cursor', 'pointer')
    .on('click', (e, c) => select('client', c))
    .on('mousemove', (e, c) => { const i = p.client.indexOf(c); V.tip(e, `client ${c}`, [['기여 비율', V.fmt.pct(p.share[i])], ['누적', V.fmt.pct(p.cum[i])]]); })
    .on('mouseleave', V.untip);
  f.g.append('path').datum(p.cum).attr('fill', 'none').attr('stroke', ENC.ink).attr('stroke-width', 1.8)
    .attr('d', d3.line().x((d, i) => x(String(p.client[i])) + x.bandwidth() / 2).y(d => y(d)));
}

function renderResource(v, el) {
  const r = v.resource, k = S.view.res, B = V.css('--seq-hi');
  if (k === 'cpu') V.line(el, { x: r.round, yLabel: 'CPU %', band: { lo: r.cpu_min, hi: r.cpu_max, color: B }, series: [{ name: '프로세스 CPU 평균 (띠 = 최소~최대)', values: r.cpu_mean, color: ENC.ink }], legend: false });
  else if (k === 'mem') V.line(el, { x: r.round, yLabel: 'RSS MB', band: { lo: r.mem_min, hi: r.mem_max, color: B }, series: [{ name: 'RSS 평균 (띠 = 최소~최대)', values: r.mem_mean, color: ENC.ink }], legend: false });
  else if (k === 'time') V.line(el, { x: r.round, yLabel: '초', series: [{ name: '라운드 계산 시간', values: r.wall_s, color: ENC.ink }] });
  else V.line(el, { x: r.round, yLabel: 'MB', series: [{ name: '라운드 통신량', values: r.bytes.map(b => b / 2 ** 20), color: ENC.ink }] });
}

function spark(td, values, color, domain) {
  const w = 64, h = 16, vals = values.map(q => q == null ? null : q);
  const svg = d3.select(td).append('svg').attr('width', w).attr('height', h);
  const def = vals.filter(q => q != null);
  if (!def.length) return;
  const x = d3.scaleLinear().domain([0, vals.length - 1 || 1]).range([1, w - 3]);
  const y = d3.scaleLinear().domain(domain || d3.extent(def)).range([h - 2, 2]);
  svg.append('path').datum(vals).attr('d', d3.line().defined(q => q != null).x((q, i) => x(i)).y(q => y(q))).attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.3);
}

function renderSparkTable(v, el, on) {
  el = V.clear(el);
  const s = v.clients.series, cnt = v.clients.counts, cl = S.snap.clients, assign = lastAssign(), st = v.clients.status;
  const tbl = document.createElement('table'); tbl.className = 'tbl';
  tbl.innerHTML = '<thead><tr><th>client</th><th>소속</th><th>전역 정확도</th><th>지연</th><th>품질 Q</th><th class="num">타임아웃</th><th class="num">이상치</th><th>최근</th></tr></thead>';
  const tb = document.createElement('tbody');
  cl.forEach((c, i) => {
    const tr = document.createElement('tr');
    const last = st[i][st[i].length - 1];
    tr.style.opacity = on && !on.has(i) ? 0.25 : 1;
    if (on && on.has(i)) tr.style.background = 'var(--surface-2)';
    tr.style.cursor = 'pointer';
    tr.onclick = () => select('client', i);
    tr.innerHTML = `<td><span class="cc" style="background:${clientColor(i)}">${i}</span>${c.malicious ? ' ×악성' : ''}</td><td>${edgeLabel(assign[i])}</td><td></td><td></td><td></td>
      <td class="num">${cnt[i].timeout}</td><td class="num"${cnt[i].outlier ? ` style="color:${ENC.alert};font-weight:700"` : ''}>${cnt[i].outlier}</td>
      <td><span class="sq" style="background:${ENC.status[last] || ENC.status.idle}"></span>${STATUS_KO[last] || last}</td>`;
    const tds = tr.querySelectorAll('td');
    spark(tds[2], s.global_acc[i], ENC.ink, [0, 1]); spark(tds[3], s.latency_ms[i], ENC.ink); spark(tds[4], s.quality[i], ENC.ink, [0, 1]);
    tb.appendChild(tr);
  });
  tbl.appendChild(tb); el.appendChild(tbl);
}

function renderSankey(v, el, on) {
  el = V.clear(el);
  const sk = v.network.sankey, fade = fadeOf(on);
  if (!sk.links.length || !d3.sankey) return V.empty(el, d3.sankey ? '데이터 없음' : 'd3-sankey 로드 실패');
  const f = V.frame(el, null, { l: 4, r: 4, t: 4, b: 4 });
  const gen = d3.sankey().nodeWidth(9).nodePadding(Math.max(1, Math.min(4, f.h / sk.nodes.length / 3))).extent([[0, 0], [f.w, f.h]]).nodeAlign(d3.sankeyJustify);
  const graph = gen({ nodes: sk.nodes.map(d => Object.assign({}, d)), links: sk.links.map(d => Object.assign({}, d)) });
  const ncol = d => d.kind === 'client' ? clientColor(d.id) : ENC.ink;
  const nOp = d => d.kind === 'client' ? fade(d.id) : d.kind === 'edge' ? (edgeOn(d.id) ? 1 : 0.2) : 1;
  f.g.append('g').selectAll('path').data(graph.links).join('path').attr('d', d3.sankeyLinkHorizontal())
    .attr('fill', 'none').attr('stroke', l => l.source.kind === 'client' ? clientColor(l.source.id) : ENC.ink).attr('stroke-opacity', l => 0.4 * Math.min(nOp(l.source), nOp(l.target)))
    .attr('stroke-width', l => Math.max(1, l.width))
    .on('mousemove', (e, l) => V.tip(e, `${l.source.name} → ${l.target.name}`, [['누적 전송량', V.fmt.mb(l.value / 2 ** 20)]])).on('mouseleave', V.untip);
  const nd = f.g.append('g').selectAll('g').data(graph.nodes).join('g').attr('opacity', nOp).style('cursor', d => d.kind === 'server' ? null : 'pointer')
    .on('click', (e, d) => { if (d.kind === 'client') select('client', d.id); else if (d.kind === 'edge') select('edge', d.id); });
  nd.append('rect').attr('x', d => d.x0).attr('y', d => d.y0).attr('width', d => d.x1 - d.x0).attr('height', d => Math.max(1, d.y1 - d.y0)).attr('fill', ncol)
    .on('mousemove', (e, d) => V.tip(e, d.kind === 'edge' ? edgeLabel(d.id) : d.name, [['총량', V.fmt.mb(d.value / 2 ** 20)]])).on('mouseleave', V.untip);
  nd.filter(d => d.kind !== 'client' || d.y1 - d.y0 >= 7).append('text').attr('x', d => d.x0 < f.w / 2 ? d.x1 + 3 : d.x0 - 3).attr('y', d => (d.y0 + d.y1) / 2 + 3)
    .attr('text-anchor', d => d.x0 < f.w / 2 ? 'start' : 'end').attr('font-size', d => d.kind === 'client' ? 8.5 : 10).attr('font-weight', 600)
    .attr('fill', d => d.kind === 'client' ? clientColor(d.id) : ENC.ink).text(d => d.kind === 'client' ? 'c' + d.id : d.kind === 'edge' ? edgeLabel(d.id) : '★서버');
}

init();
