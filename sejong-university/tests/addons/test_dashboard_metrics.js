const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const root = path.resolve(__dirname, '..', '..', '1_simulation', '1-2_fl_visualization', 'src');
const js = fs.readFileSync(path.join(root, 'defai_addons/monitor/static/dashboard.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'defai_addons/monitor/templates/dashboard.html'), 'utf8');
const start = js.indexOf('function renderLdflCard(el) {');
const end = js.indexOf(String.fromCharCode(10) + "function ", start + 1);
const source = js.slice(start, end);
assert(start >= 0 && end > start);
assert(js.includes("$('ldflMetric').onchange = () => renderLdflCard($('ldflChart'))"));
const select = { value: 'f1' };
const el = { classList: { add(){}, remove(){} }, innerHTML: '' };
let captured;
const metrics = {f1:.4, precision:.6, recall:.8, accuracy:.9};
const local = {f1:.3, precision:.5, recall:.7, accuracy:.85};
const state = {view:{ldfl:'f1'}, job:{config:{envs:['fl'],epochs:2},labels:{fl:'FL'},
  progress:{FL:[metrics,local]},report:{results:[{key:'fl',name:'FL',global:metrics,local_mean:local,
    cost:{raw_bytes:0,model_bytes:20,stat_bytes:5},privacy:{raw_data_leaves_site:false,dp:null}}]}}};
const context = {
  S:state, $:id=>id==='ldflMetric'?select:el, ENV_SHORT:{}, ENV_DASH:[''], ENC:{ink:'#222'},
  badge:()=>'',
  d3:{max:(xs,f)=>Math.max(...xs.map(f)),range:(a,b)=>Array.from({length:b-a},(_,i)=>a+i),format:()=>x=>x},
  V:{empty:(_,message)=>{captured={kind:'empty',message}},clear:x=>x,
    line:(_,cfg)=>{captured={kind:'line',cfg}},bars:(_,cfg)=>{captured={kind:'bars',cfg}},fmt:{mb:x=>x,f1:x=>x}}
};
vm.createContext(context);
vm.runInContext(source,context);
const names={f1:'F1',precision:'정밀도',recall:'재현율',accuracy:'정확도'};
for(const metric of Object.keys(names)) {
  assert(html.includes(`value="${metric}"`));
  select.value=metric;
  state.view.ldfl='f1';
  context.renderLdflCard(el);
  assert.equal(captured.kind,'bars');
  assert.deepEqual(Array.from(captured.cfg.values[0]),[metrics[metric],local[metric]]);
  assert(captured.cfg.keys[0].includes(names[metric]));
  state.view.ldfl='prog';
  context.renderLdflCard(el);
  assert.equal(captured.kind,'line');
  assert.deepEqual(Array.from(captured.cfg.series[0].values),[metrics[metric],local[metric]]);
}
state.view.ldfl='cost';
context.renderLdflCard(el);
assert(el.innerHTML.includes('원 데이터 이동'));
select.value='invalid';state.view.ldfl='f1';context.renderLdflCard(el);
assert.equal(captured.cfg.values[0][0],metrics.f1);
state.job.report=null;select.value='recall';context.renderLdflCard(el);
assert.equal(captured.kind,'line');assert.equal(captured.cfg.series[0].values[0],.8);
console.log('PASS: 4 metrics, global/local values, live trends, costs and invalid-input fallback');
