"""Inspect the offline design document in a dedicated headless Edge instance."""
import asyncio
import base64
import csv
import itertools
import json
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
HTML = REPORT / 'HbA1c_FL_20만명_세부계산설계.html'
QA = ROOT / 'benchmarks' / 'design_html_qa'
QA.mkdir(exist_ok=True)
soup = BeautifulSoup(HTML.read_text(encoding='utf-8'), 'html.parser')
config = json.loads(soup.find(id='design-config').string)
(REPORT / 'HbA1c_FL_20만명_기본설정.json').write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
matrix = config['experiment_matrix']
run_path = REPORT / 'HbA1c_FL_20만명_주실험목록.csv'
with run_path.open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['run_id','health_beta','lab_attendance','phi','method','model','K','seed','status'])
    axes = [config['behavior']['health_beta_grid'], matrix['attendance_grid'],
            config['generator']['phi_grid'], matrix['methods'], matrix['models'],
            matrix['completed_caps'], matrix['seeds']]
    for count, row in enumerate(itertools.product(*axes), 1):
        writer.writerow([f'planned_{count:05}', *row, 'planned_not_executed'])
assert count == 4860
ids = [node['id'] for node in soup.select('[id]')]
assert len(ids) == len(set(ids)), 'duplicate IDs'
for link in soup.select('a[href^="#"]'):
    assert link['href'][1:] in ids, f'broken anchor {link}'
for link in soup.select('a[href]'):
    href = link['href']
    if not href.startswith(('#','https://','http://')):
        assert (REPORT / href).exists(), f'broken local link {href}'
assert not soup.select('script[src],link[rel="stylesheet"],img[src^="http"]')
assert len(config['features']) == 20
assert all(abs(sum(row)-1) < 1e-12 for row in config['behavior']['transition_matrix'])
assert all(sum((a+1)*b for a,b in zip(m['architecture'][:-1],m['architecture'][1:]))
           == m['parameters'] for m in config['models'])


async def verify():
    targets = json.load(urllib.request.urlopen('http://127.0.0.1:9337/json'))
    target = next(t for t in targets if t['type'] == 'page')
    async with connect(target['webSocketDebuggerUrl'], max_size=30_000_000) as ws:
        serial = 0
        runtime_errors = []

        async def rpc(method, params=None):
            nonlocal serial
            serial += 1
            expected = serial
            await ws.send(json.dumps({'id': expected,'method': method,'params':params or {}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get('method') == 'Runtime.exceptionThrown':
                    runtime_errors.append(msg['params'])
                if msg.get('id') == expected:
                    if 'error' in msg:
                        raise RuntimeError(msg['error'])
                    return msg.get('result', {})

        async def evaluate(expression):
            result = await rpc('Runtime.evaluate', {'expression':expression,'returnByValue':True})
            if 'exceptionDetails' in result:
                raise RuntimeError(result['exceptionDetails'])
            return result.get('result', {}).get('value')

        async def viewport(width, height):
            await rpc('Emulation.setDeviceMetricsOverride', {
                'width':width,'height':height,'deviceScaleFactor':1,'mobile':width<600})
            await asyncio.sleep(.15)

        async def shot(filename, anchor=None):
            await evaluate('document.documentElement.style.scrollBehavior="auto"')
            if anchor:
                await evaluate(f'document.getElementById({json.dumps(anchor)}).scrollIntoView({{behavior:"instant",block:"start"}})')
            else:
                await evaluate('window.scrollTo(0,0)')
            await asyncio.sleep(.25)
            result = await rpc('Page.captureScreenshot', {'format':'png','captureBeyondViewport':False})
            (QA / filename).write_bytes(base64.b64decode(result['data']))

        await rpc('Runtime.enable')
        await rpc('Page.enable')
        await viewport(1440, 1100)
        await rpc('Page.navigate', {'url':HTML.as_uri()})
        for _ in range(30):
            if await evaluate('document.documentElement.dataset.designReady === "true"'):
                break
            await asyncio.sleep(.1)
        assert await evaluate('document.documentElement.dataset.designReady === "true"')
        default = await evaluate('({kernel: +document.getElementById("capacity").dataset.kernelSeconds, sparse:+document.getElementById("capacity").dataset.sparseBytes, sections:document.querySelectorAll("main section").length})')
        assert abs(default['kernel'] - .0276648*410) < .00001
        assert abs(default['sparse'] - 134400000) < .01
        assert default['sections'] == 23
        actual = await evaluate('({...document.getElementById("optimization-results").dataset})')
        assert int(actual['population']) == 200000
        assert int(actual['testModels']) == 25
        final = json.loads((ROOT / 'experiments/fl_comparison/results/main/final_results.json').read_text(encoding='utf-8'))
        assert len(final['runs']) == 25
        assert len(set(r['seed'] for r in final['runs'])) == 5
        assert all(0 < r['actual_aggregations'] <= 410 for r in final['runs'])
        assert int(actual['trainingCells']) == final['manifest']['candidate_count_per_seed'] * 5
        await shot('01_desktop_overview.png')
        await shot('02_behavior.png', 'behavior')
        await shot('03_health_formulas.png', 'health')
        await shot('04_capacity_controls.png', 'capacity')
        await shot('05_capacity_chart.png', 'memory-chart')
        await shot('06_benchmark_chart.png', 'benchmark-chart')
        await shot('11_comparison_performance.png', 'comparison-performance')
        await shot('12_comparison_paired.png', 'comparison-paired')
        await shot('13_comparison_convergence.png', 'comparison-convergence')
        await evaluate('document.getElementById("c-cgm").value=100;document.getElementById("c-coverage").value=100;document.getElementById("c-coverage").dispatchEvent(new Event("input"))')
        cgm_full = await evaluate('document.getElementById("out-cgm").textContent')
        assert cgm_full == '168.19 GB', cgm_full
        await evaluate('document.getElementById("c-n").value=100000;document.getElementById("c-n").dispatchEvent(new Event("input"))')
        cgm_half = await evaluate('document.getElementById("out-cgm").textContent')
        assert cgm_half == '84.1 GB', cgm_half
        await evaluate('document.getElementById("c-k").value=200000;document.getElementById("c-k").dispatchEvent(new Event("input"))')
        assert 'N으로 제한' in await evaluate('document.getElementById("calc-note").textContent')
        await rpc('Page.reload')
        await asyncio.sleep(.3)
        await rpc('Browser.setDownloadBehavior', {'behavior':'allow','downloadPath':str(QA)})
        downloaded = QA / 'hba1c-fl-simulation-design-v1.json'
        downloaded.unlink(missing_ok=True)  # A derived QA download, not the report configuration.
        await evaluate('document.getElementById("download-config").click()')
        for _ in range(30):
            if downloaded.exists():
                break
            await asyncio.sleep(.1)
        assert json.loads(downloaded.read_text(encoding='utf-8')) == config
        await viewport(390, 900)
        await shot('07_mobile_overview.png')
        mobile_overflow = await evaluate('({page:document.documentElement.scrollWidth,viewport:innerWidth})')
        assert mobile_overflow['page'] <= mobile_overflow['viewport'], mobile_overflow
        await shot('08_mobile_capacity.png', 'capacity')
        await shot('09_mobile_memory_chart.png', 'memory-chart')
        await shot('10_mobile_benchmark.png', 'benchmark-chart')
        await shot('14_mobile_comparison.png', 'comparison-performance')
        await viewport(1440,1100)
        await rpc('Emulation.setEmulatedMedia', {'media':'print'})
        pdf = await rpc('Page.printToPDF', {'printBackground':True,'preferCSSPageSize':True})
        (QA / 'design_print_preview.pdf').write_bytes(base64.b64decode(pdf['data']))
        assert not runtime_errors, runtime_errors
        report = {'html':str(HTML),'static_ids_unique':True,'anchor_links_valid':True,
                  'external_runtime_dependencies':False,'sections':default['sections'],
                  'configuration_features':len(config['features']),'planned_runs':count,
                  'default_kernel_seconds':default['kernel'],'default_sparse_bytes':default['sparse'],
                  'calculator_full_cgm':cgm_full,'calculator_half_population_cgm':cgm_half,
                  'K_clamping_verified':True,'configuration_download_verified':True,
                  'mobile_page_overflow':mobile_overflow,
                  'runtime_exceptions':runtime_errors,'screenshots':14,
                  'training_engine_executed':True,'actual_selected_models':25,
                  'finalized_training_cells':int(actual['trainingCells'])}
        (QA / 'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))


asyncio.run(verify())
