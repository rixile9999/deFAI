const V = (() => {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const series = i => css('--s' + ((i % 8) + 1));
  const status = s => ({ ok: css('--good'), good: css('--good'), timeout: css('--warning'), degraded: css('--warning'),
    warn: css('--warning'), outlier: css('--critical'), down: css('--critical'), alert: css('--critical'),
    critical: css('--critical'), idle: css('--muted'), info: css('--muted') }[s] || css('--muted'));
  const STATUS_KO = { ok: '정상', timeout: '타임아웃', degraded: '저하', down: '중단', idle: '미참여', outlier: '이상치' };
  const fmt = { pct: d3.format('.1%'), f3: d3.format('.3f'), f2: d3.format('.2f'), f1: d3.format('.1f'),
    int: d3.format(',d'), mb: v => d3.format('.2f')(v) + ' MB', auto: v => (v == null ? '–' : Math.abs(v) >= 100 ? d3.format(',.0f')(v) : d3.format('.3~f')(v)) };

  const tipEl = () => document.getElementById('tip');
  function tip(evt, title, rows) {
    const t = tipEl();
    t.innerHTML = (title ? `<div class="t">${title}</div>` : '') +
      (rows || []).map(([k, v, c]) => `<div class="r"><span class="k">${c ? `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${c};margin-right:5px"></span>` : ''}${k}</span><span>${v}</span></div>`).join('');
    t.style.display = 'block';
    const w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - 14;
    t.style.left = x + 'px'; t.style.top = y + 'px';
  }
  const untip = () => { tipEl().style.display = 'none'; };

  const cursors = new Map();
  const cursor = (el, fn) => cursors.set(el, fn);
  function roundAt(r, src) { cursors.forEach((fn, el) => { if (!el.isConnected) return cursors.delete(el); if (el !== src) fn(r); }); }

  function clear(el) { el = typeof el === 'string' ? document.getElementById(el) : el; cursors.delete(el); el.innerHTML = ''; return el; }
  function empty(el, msg) { el = clear(el); el.innerHTML = `<div class="empty">${msg || '데이터 없음'}</div>`; return el; }

  function swatch(it) {
    const c = it.color || 'var(--ink)';
    if (it.type === 'ln') return `<svg class="lg" width="16" height="8"><line x1="0" y1="4" x2="16" y2="4" stroke="${c}" stroke-width="${it.width || 2.4}" stroke-linecap="round"/></svg>`;
    if (it.type === 'dot') return `<svg class="lg" width="10" height="10"><circle cx="5" cy="5" r="4" fill="${c}" ${it.border ? 'stroke="var(--muted)" stroke-width="1"' : ''}/></svg>`;
    return `<span class="sw" style="background:${c};${it.border ? 'box-shadow:inset 0 0 0 1px var(--axis)' : ''}"></span>`;
  }
  function legend(el, items) {
    const d = document.createElement('div');
    d.className = 'legend';
    d.innerHTML = items.map(it => `<span>${swatch(it)}${it.name}</span>`).join('');
    el.appendChild(d);
    return d;
  }

  function avail(el, min) {
    const used = [...el.children].reduce((a, c) => a + c.offsetHeight, 0);
    return Math.max(min || 90, (el.clientHeight || 260) - used - 4);
  }

  function frame(el, height, m) {
    m = Object.assign({ t: 8, r: 16, b: 26, l: 44 }, m || {});
    if (!height || height === 'fill') height = avail(el);
    const W = Math.max(160, el.clientWidth || 600);
    const svg = d3.select(el).append('svg').attr('width', W).attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${m.l},${m.t})`);
    return { svg, g, w: W - m.l - m.r, h: height - m.t - m.b, m };
  }

  const intTicks = (x, n) => { const [a, b] = x.domain(); return Number.isInteger(a) && Number.isInteger(b) ? Math.max(1, Math.min(n, b - a)) : n; };

  function axes(f, x, y, opts) {
    opts = opts || {};
    f.g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(opts.yTicks || 5).tickSize(-f.w).tickFormat(''));
    f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`)
      .call(d3.axisBottom(x).ticks(intTicks(x, opts.xTicks || Math.min(10, Math.floor(f.w / 60)))).tickSizeOuter(0).tickFormat(opts.xFmt || null));
    f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(opts.yTicks || 5).tickSizeOuter(0).tickFormat(opts.yFmt || null));
    if (opts.yLabel) f.g.append('text').attr('x', -f.m.l + 2).attr('y', -8).attr('font-size', 10.5).attr('fill', 'var(--muted)').text(opts.yLabel);
    if (opts.xLabel) f.g.append('text').attr('x', f.w).attr('y', f.h + 24).attr('text-anchor', 'end').attr('font-size', 10.5).text(opts.xLabel);
  }

  function line(el, opts) {
    el = clear(el);
    const S = opts.series.filter(s => s.values && s.values.some(v => v != null));
    if (!opts.x || !opts.x.length || !S.length) return empty(el);
    if (opts.legendItems) legend(el, opts.legendItems);
    else if (opts.legend !== false && S.length > 1) legend(el, S.map(s => ({ name: s.name, color: s.color, type: 'ln' })));
    const f = frame(el, opts.height, { r: opts.directLabels ? 70 : 8, l: 36, t: opts.yLabel ? 18 : 8 });
    const x = d3.scaleLinear().domain(d3.extent(opts.x)).range([0, f.w]);
    let all = S.flatMap(s => s.values).filter(v => v != null);
    if (opts.band) all = all.concat(opts.band.lo.filter(v => v != null), opts.band.hi.filter(v => v != null));
    const y = d3.scaleLinear().domain(opts.yDomain || [Math.min(0, d3.min(all)), d3.max(all) * 1.05 || 1]).nice().range([f.h, 0]);
    f.x = x; f.y = y;
    axes(f, x, y, opts);
    if (opts.under) opts.under(f, x, y);  // 선 아래에 그릴 요소 (seed 범위 띠 등)
    if (opts.band) {
      const ar = d3.area().defined((d, i) => opts.band.lo[i] != null).x((d, i) => x(opts.x[i]))
        .y0((d, i) => y(opts.band.lo[i])).y1((d, i) => y(opts.band.hi[i]));
      f.g.append('path').datum(opts.x).attr('d', ar).attr('fill', opts.band.color).attr('opacity', 0.18);
    }
    S.forEach(s => {
      const ln = d3.line().defined(v => v != null).x((v, i) => x(opts.x[i])).y(v => y(v));
      const p = f.g.append('path').datum(s.values).attr('d', ln).attr('fill', 'none').attr('stroke', s.color)
        .attr('stroke-width', s.width || 2).attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('opacity', s.opacity ?? 1);
      Object.entries(s.attrs || {}).forEach(([k, v]) => p.attr(k, v));
      if (opts.directLabels) {
        let k = s.values.length - 1; while (k > 0 && s.values[k] == null) k--;
        f.g.append('text').attr('x', x(opts.x[k]) + 6).attr('y', y(s.values[k]) + 4).attr('fill', 'var(--ink-2)').text(s.name);
      }
    });
    if (opts.over) opts.over(f, x, y);  // 선 위에 그릴 요소 (강조될 때만 보이는 클라이언트 선 등)
    (opts.markers || []).forEach(mk => {
      f.g.append('circle').attr('r', 4).attr('cx', x(mk.x)).attr('cy', y(mk.y))
        .attr('fill', mk.color || status('alert')).attr('stroke', 'var(--surface)').attr('stroke-width', 1.5)
        .on('mousemove', e => tip(e, `라운드 ${mk.x}`, [[mk.label, '']])).on('mouseleave', untip);
    });
    const cross = f.g.append('line').attr('y1', 0).attr('y2', f.h).attr('stroke', 'var(--ink-2)').attr('stroke-opacity', 0.45).attr('pointer-events', 'none').style('display', 'none');
    const linked = opts.roundLink !== false;
    if (linked) cursor(el, r => { const i = r == null ? -1 : opts.x.indexOf(r); if (i < 0) cross.style('display', 'none'); else cross.attr('x1', x(r)).attr('x2', x(r)).style('display', null); });
    const dots = S.map(s => f.g.append('circle').attr('r', 4).attr('fill', s.color).attr('stroke', 'var(--surface)').attr('stroke-width', 2).style('display', 'none'));
    const fv = opts.yFmt || fmt.auto;
    f.g.append('rect').attr('class', 'lc-hit').attr('width', f.w).attr('height', f.h).attr('fill', 'transparent')
      .on('mousemove', e => {
        const [mx] = d3.pointer(e);
        const xv = x.invert(mx);
        const i = d3.leastIndex(opts.x, v => Math.abs(v - xv));
        cross.attr('x1', x(opts.x[i])).attr('x2', x(opts.x[i])).style('display', null);
        S.forEach((s, k) => { const v = s.values[i]; dots[k].style('display', v == null ? 'none' : null).attr('cx', x(opts.x[i])).attr('cy', v == null ? 0 : y(v)); });
        const rows = S.map(s => [s.name, s.values[i] == null ? '–' : fv(s.values[i]), s.color]);
        if (opts.band) rows.push(['범위', `${fv(opts.band.lo[i])} – ${fv(opts.band.hi[i])}`]);
        tip(e, `${opts.xName || '라운드'} ${opts.x[i]}`, rows);
        if (linked) roundAt(opts.x[i], el);
      })
      .on('mouseleave', () => { cross.style('display', 'none'); dots.forEach(d => d.style('display', 'none')); untip(); if (linked) roundAt(null, el); })
      .lower();
    return f;
  }

  function bars(el, opts) {
    el = clear(el);
    if (!opts.groups.length) return empty(el);
    if (opts.keys.length > 1) legend(el, opts.keys.map((k, i) => ({ name: k, color: opts.colors[i] })));
    const f = frame(el, opts.height, { b: opts.rotate ? 70 : 20, l: 34, r: 4, t: 10 });
    const x0 = d3.scaleBand().domain(opts.groups).range([0, f.w]).paddingInner(0.25).paddingOuter(0.05);
    const x1 = d3.scaleBand().domain(opts.keys).range([0, x0.bandwidth()]).paddingInner(0.12);
    const all = opts.values.flat().filter(v => v != null);
    const y = d3.scaleLinear().domain(opts.yDomain || [0, d3.max(all) * 1.1 || 1]).nice().range([f.h, 0]);
    f.g.append('g').attr('class', 'gridline').call(d3.axisLeft(y).ticks(5).tickSize(-f.w).tickFormat(''));
    const xa = f.g.append('g').attr('class', 'axis').attr('transform', `translate(0,${f.h})`).call(d3.axisBottom(x0).tickSizeOuter(0));
    if (opts.rotate) xa.selectAll('text').attr('transform', 'rotate(-25)').attr('text-anchor', 'end');
    f.g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(5).tickFormat(opts.yFmt || null));
    const fv = opts.yFmt || fmt.auto;
    opts.groups.forEach((g, gi) => {
      const gg = f.g.append('g').attr('transform', `translate(${x0(g)},0)`);
      opts.keys.forEach((k, ki) => {
        const v = opts.values[gi][ki]; if (v == null) return;
        const bw = x1.bandwidth(), y0 = y(Math.max(0, y.domain()[0]));
        const h = Math.max(0, y0 - y(v)), r = Math.min(2, bw / 2, h);
        const px = x1(k);
        gg.append('path').attr('d', `M${px},${y0}V${y0 - h + r}Q${px},${y0 - h} ${px + r},${y0 - h}H${px + bw - r}Q${px + bw},${y0 - h} ${px + bw},${y0 - h + r}V${y0}Z`)
          .attr('fill', opts.colors[ki])
          .on('mousemove', e => tip(e, g, [[k, fv(v), opts.colors[ki]]])).on('mouseleave', untip);
        if (opts.labels !== false && bw > 18) gg.append('text').attr('fill', 'var(--ink)').attr('x', px + bw / 2).attr('y', y0 - h - 4).attr('text-anchor', 'middle').attr('font-size', 10).text(fv(v));
      });
    });
    return f;
  }

  const seq = domain => d3.scaleLinear().domain(domain || [0, 1]).range([css('--seq-lo'), css('--seq-hi')]).interpolate(d3.interpolateLab).clamp(true);
  const div = domain => { const d = domain || [-1, 0, 1]; return d3.scaleLinear().domain([d[0], d[d.length - 1]]).range([css('--seq-lo'), css('--seq-hi')]).interpolate(d3.interpolateLab).clamp(true); };

  function colorbar(g, scale, x, y, w, label, f) {
    const id = 'cb' + Math.random().toString(36).slice(2, 8);
    const grad = g.append('defs').append('linearGradient').attr('id', id);
    const [a, b] = [scale.domain()[0], scale.domain()[scale.domain().length - 1]];
    d3.range(0, 1.01, 0.1).forEach(t => grad.append('stop').attr('offset', t).attr('stop-color', scale(a + (b - a) * t)));
    g.append('rect').attr('x', x).attr('y', y).attr('width', w).attr('height', 8).attr('rx', 2).attr('fill', `url(#${id})`);
    g.append('text').attr('x', x).attr('y', y + 20).attr('font-size', 10).text((f || fmt.auto)(a));
    g.append('text').attr('x', x + w).attr('y', y + 20).attr('text-anchor', 'end').attr('font-size', 10).text((f || fmt.auto)(b));
    if (label) g.append('text').attr('x', x + w / 2).attr('y', y + 20).attr('text-anchor', 'middle').attr('font-size', 10).text(label);
  }

  return { swatch, cursor, roundAt, avail, intTicks, css, series, status, STATUS_KO, fmt, tip, untip, clear, empty, legend, frame, axes, line, bars, seq, div, colorbar };
})();
