'use strict';
const $ = (selector) => document.querySelector(selector);
const names = {
  dummy_prior: 'Константный прогноз · срез',
  logistic_regression: 'Логистическая · срез',
  random_forest: 'Случайный лес · срез',
  history_logistic_regression: 'Логистическая · история',
  history_random_forest: 'Случайный лес · история',
};
const percent = (x) => x == null ? '—' : (100 * x).toFixed(1) + '%';
const dec = (x) => x == null ? '—' : Number(x).toFixed(3);
let predictionRows = [];

function addTextCell(tr, value, className='') {
  const cell = document.createElement('td');
  cell.textContent = String(value);
  if (className) cell.className = className;
  tr.appendChild(cell);
  return cell;
}
function renderPredictions() {
  const threshold = Number($('#threshold').value) / 100;
  $('#threshold-display').textContent = Math.round(threshold * 100) + '%';
  const q = $('#search').value.toUpperCase().trim();
  const matched = predictionRows.filter(r => r.risk_probability >= threshold && r.student_id.toUpperCase().includes(q));
  $('#results-note').textContent = `${matched.length ? 'Подходят под фильтр: '+matched.length+' из '+predictionRows.length+' синтетических наблюдений' : 'Совпадений нет. Уменьшите порог или измените ID.'}`;
  $('#prediction-rows').replaceChildren();
  matched.slice(0, 25).forEach(r => {
    const tr = document.createElement('tr');
    const cell = document.createElement('td');
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'id-link'; button.textContent = r.student_id;
    button.title = 'Показать историю студента';
    button.addEventListener('click', () => {
      $('#student-id').value = r.student_id;
      showStudent(r.student_id);
      location.hash = 'history'; $('#history').scrollIntoView({behavior:'smooth'});
    });
    cell.append(button); tr.append(cell);
    addTextCell(tr, Number(r.current_avg_grade).toFixed(2));
    const risk = r.risk_probability >= 0.65 ? 'high' : r.risk_probability >= 0.4 ? 'mid' : 'low';
    addTextCell(tr, percent(r.risk_probability), `risk ${risk}`);
    addTextCell(tr, r.next_term_low_performance ? '1 · ниже 3,0' : '0 · не ниже 3,0');
    $('#prediction-rows').appendChild(tr);
  });
}
function renderSummary(s) {
  $('#test-count').textContent = s.counts.test;
  $('#model-choice').textContent = names[s.selected_model] || s.selected_model;
  $('#ap-value').textContent = dec(s.metrics[s.selected_model].test.average_precision);
  $('#auc-value').textContent = dec(s.metrics[s.selected_model].test.roc_auc);
  $('#model-rows').replaceChildren();
  $('#bars').replaceChildren();
  for (const [key, metric] of Object.entries(s.metrics)) {
    const m = metric.test, tr = document.createElement('tr');
    addTextCell(tr, (names[key] || key) + (s.selected_model === key ? ' · выбрана' : ''), s.selected_model === key ? 'best' : '');
    ['average_precision','roc_auc','recall','f1','brier'].forEach(k => addTextCell(tr,dec(m[k])));
    $('#model-rows').appendChild(tr);
    const row = document.createElement('div');row.className = 'barrow' + (key === s.selected_model ? ' selected' : '');
    const label = document.createElement('span');label.textContent = names[key] || key;
    const track = document.createElement('div');track.className = 'bartrack';
    const bar = document.createElement('div');bar.className = 'barfill';
    bar.style.width = Math.max(0,Math.min(1,m.average_precision || 0)) * 100 + '%';
    track.appendChild(bar);
    const value = document.createElement('strong');value.textContent = dec(m.average_precision);
    row.append(label,track,value);$('#bars').appendChild(row);
  }
  $('#comparison-rows').replaceChildren();
  for (const key of ['logistic_regression','random_forest']) {
    const noHistory = s.metrics[key].test;
    const withHistory = s.metrics['history_' + key].test;
    const delta = s.history_vs_snapshot[key].test;
    const tr = document.createElement('tr');
    addTextCell(tr, key === 'logistic_regression' ? 'Логистическая регрессия' : 'Случайный лес');
    addTextCell(tr,dec(noHistory.average_precision));
    addTextCell(tr,dec(withHistory.average_precision));
    addTextCell(tr,(delta.average_precision >= 0 ? '+' : '') + dec(delta.average_precision));
    addTextCell(tr,(delta.brier >= 0 ? '+' : '') + dec(delta.brier));
    $('#comparison-rows').appendChild(tr);
  }
  $('#stats').hidden = false;
}

const svgNS = 'http://www.w3.org/2000/svg';
function svgElement(tag, attrs, text) {
  const node = document.createElementNS(svgNS,tag);
  for (const [key,val] of Object.entries(attrs || {})) node.setAttribute(key,String(val));
  if (text !== undefined) node.textContent = text;
  return node;
}
function renderLineChart(container, rows, field, min, max, color, asPercent=false) {
  container.replaceChildren();
  if (!rows.length) return;
  const W=560, H=238, left=48, right=18, top=18, bottom=35;
  const x = (i) => left + (rows.length === 1 ? 0.5 : i/(rows.length-1)) * (W-left-right);
  const y = (v) => top + (max-v)/(max-min)*(H-top-bottom);
  const svg = svgElement('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':asPercent?'График посещаемости по периодам':'График средней оценки по периодам'});
  for (let i=0;i<=4;i++) {
    const v = min+(max-min)*i/4;
    const yy = y(v);
    svg.append(svgElement('line',{x1:left,y1:yy,x2:W-right,y2:yy,stroke:'#e6eaf4','stroke-width':1}));
    svg.append(svgElement('text',{x:left-9,y:yy+4,'text-anchor':'end',fill:'#77849d','font-size':12},asPercent?`${Math.round(v*100)}%`:v.toFixed(1)));
  }
  const path = rows.map((r,i) => `${i?'L':'M'} ${x(i).toFixed(1)} ${y(Number(r[field])).toFixed(1)}`).join(' ');
  svg.append(svgElement('path',{d:path,fill:'none',stroke:color,'stroke-width':3,'stroke-linecap':'round','stroke-linejoin':'round'}));
  rows.forEach((r,i) => {
    const val = Number(r[field]);
    svg.append(svgElement('circle',{cx:x(i),cy:y(val),r:5,fill:color,stroke:'#fff','stroke-width':2}));
    svg.append(svgElement('text',{x:x(i),y:H-9,'text-anchor':'middle',fill:'#77849d','font-size':12},'Сем. '+r.term_index));
  });
  container.appendChild(svg);
}
let trajectoryRequest = 0;
async function showStudent(id) {
  const requestId = ++trajectoryRequest;
  const status = $('#trajectory-loading');
  status.hidden=false;status.textContent='Загрузка истории…';
  $('#trajectory-area').hidden=true;
  try {
    if (!/^SYN-\d+$/.test(id)) throw new Error('Введите ID вида SYN-0001');
    const response = await fetch('/api/students/'+encodeURIComponent(id)+'/trajectory');
    if (!response.ok) throw new Error(response.status === 404 ? 'Синтетический ID не найден' : 'Не удалось получить историю');
    const data = await response.json();
    if (requestId !== trajectoryRequest) return;
    const rows = data.observations;
    if (!rows.length) throw new Error('История отсутствует');
    $('#student-grade').textContent = Number(rows.at(-1).current_avg_grade).toFixed(2);
    const change = rows.at(-1).current_avg_grade-rows[0].current_avg_grade;
    $('#student-change').textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} балла`;
    $('#student-risk').textContent = data.forecast ? percent(data.forecast.risk_probability) : '—';
    renderLineChart($('#grade-chart'),rows,'current_avg_grade',2,5,'#5667ed');
    renderLineChart($('#attendance-chart'),rows,'attendance_rate',0,1,'#0c9e8e',true);
    $('#trajectory-rows').replaceChildren();
    rows.forEach(r => {
      const tr=document.createElement('tr');
      addTextCell(tr,r.term_index);
      addTextCell(tr,Number(r.current_avg_grade).toFixed(2));
      addTextCell(tr,percent(r.attendance_rate));
      addTextCell(tr,percent(r.assignment_completion_rate));
      addTextCell(tr,percent(r.credits_completion_rate));
      $('#trajectory-rows').appendChild(tr);
    });
    status.hidden=true;$('#trajectory-area').hidden=false;
  } catch (err) {
    if (requestId === trajectoryRequest) status.textContent='Ошибка: '+err.message;
  }
}
async function showCohort() {
  const response = await fetch('/api/term-trends');
  if (!response.ok) throw new Error('Нет данных для графика');
  const result = await response.json();
  renderLineChart($('#cohort-chart'),result.rows,'mean_grade',2,5,'#5667ed');
  $('#cohort-caption').textContent = `Периоды: ${result.rows.length}. Количество наблюдений в первом периоде: ${result.rows[0]?.n_students ?? '—'}. Нельзя интерпретировать синтетическую динамику как реальные закономерности.`;
}
async function init() {
  try {
    const [rSummary,rPreds] = await Promise.all([fetch('/api/summary'),fetch('/api/predictions')]);
    if (!rSummary.ok || !rPreds.ok) throw new Error('Нет результатов обучения. Запустите команды из README.md.');
    const summary = await rSummary.json();
    const preds = await rPreds.json();
    if (summary.mode !== 'SYNTHETIC_DEMO_ONLY' || preds.mode !== 'SYNTHETIC_DEMO_ONLY' || !summary.history_vs_snapshot)
      throw new Error('Необходимо пересчитать эксперимент для версии 0.2.');
    predictionRows=preds.rows;
    renderSummary(summary);renderPredictions();
    $('#loading').hidden=true;
  } catch (err) { $('#loading').hidden=false; $('#loading').textContent = 'Не удалось загрузить эксперимент: '+err.message; $('#results-note').textContent = 'Прогнозы недоступны. Обновите страницу после восстановления данных.'; }
  await Promise.allSettled([showStudent($('#student-id').value.trim().toUpperCase()),showCohort().catch(error=>{ $('#cohort-caption').textContent='Не удалось загрузить график: '+error.message; })]);
}
$('#threshold').addEventListener('input',renderPredictions);
$('#search').addEventListener('input',renderPredictions);
$('#load-student').addEventListener('click',() => showStudent($('#student-id').value.trim().toUpperCase()));
$('#student-id').addEventListener('keydown',e => {if(e.key === 'Enter') showStudent($('#student-id').value.trim().toUpperCase());});
init();


let ablationData = null;
function displayAblation() {
  if (!ablationData) return;
  const model=$('#ablation-model').value, pattern=$('#ablation-pattern').value;
  const rate=Number($('#ablation-rate').value);
  $('#ablation-pattern').disabled = rate === 0;
  // The zero-mask control is shared across both pattern labels.
  const patternUsed = rate === 0 ? 'mcar_cells' : pattern;
  const items=ablationData.rows.filter(r => r.model===model && r.pattern===patternUsed && Number(r.missing_rate)===rate)
    .sort((a,b)=>a.window_semesters-b.window_semesters);
  const selection=ablationData.window_selections.find(s=>s.model===model && s.pattern===patternUsed && Number(s.missing_rate)===rate);
  $('#ablation-rows').replaceChildren();$('#ablation-bars').replaceChildren();
  for (const r of items) {
    const tr=document.createElement('tr'), chosen=selection && r.window_semesters===selection.chosen_window_on_validation;
    addTextCell(tr,r.window_semesters+' сем.',chosen?'best':'');
    for (const key of ['validation_average_precision_mean','test_average_precision_mean','test_roc_auc_mean','test_brier_mean','test_average_precision_std'])
      addTextCell(tr,dec(r[key]));
    $('#ablation-rows').appendChild(tr);
    const row=document.createElement('div');row.className='barrow'+(chosen?' selected':'');
    const label=document.createElement('span');label.textContent=r.window_semesters+' семестров';
    const track=document.createElement('div');track.className='bartrack';
    const fill=document.createElement('div');fill.className='barfill';
    fill.style.width = Math.min(100,Math.max(0,(r.test_average_precision_mean||0)*100))+'%';
    track.appendChild(fill);
    const value=document.createElement('strong');value.textContent=dec(r.test_average_precision_mean);
    row.append(label,track,value);$('#ablation-bars').appendChild(row);
  }
  $('#ablation-summary').textContent=selection
    ? `Окно, выбранное по валидации: ${selection.chosen_window_on_validation} сем. Тест AP: ${dec(selection.test_ap_chosen)}; срез (1 сем.): ${dec(selection.test_ap_snapshot)}; Δ AP: ${selection.test_ap_delta_vs_snapshot >=0?'+':''}${dec(selection.test_ap_delta_vs_snapshot)}. Данные синтетические.`
    : 'Для выбранного условия результаты отсутствуют.';
}
async function loadAblation() {
  try {
    const response = await fetch('/api/history-ablation');
    if(!response.ok) throw new Error('Запустите эксперимент v0.3 по README.md');
    ablationData=await response.json();
    if(ablationData.mode!=='SYNTHETIC_DEMO_ONLY') throw new Error('Ожидались только синтетические данные');
    displayAblation();
  } catch(e) { $('#ablation-summary').textContent='Эксперимент недоступен: '+e.message; }
}
for (const name of ['ablation-model','ablation-pattern','ablation-rate'])
  $('#'+name).addEventListener('change',displayAblation);
loadAblation();


// v0.4: comparison of validation-selected fixed window and a per-row quality gate.
let adaptiveData = null;
function renderAdaptive() {
  if (!adaptiveData) return;
  const model = $('#adaptive-model').value;
  const rate = Number($('#adaptive-rate').value);
  $('#adaptive-pattern').disabled = rate === 0;
  const pattern = rate === 0 ? 'mcar_cells' : $('#adaptive-pattern').value;
  const key = x => x.model === model && x.pattern === pattern && Number(x.missing_rate) === rate;
  const result = adaptiveData.comparison.find(key);
  const selected = adaptiveData.selections.find(key);
  $('#adaptive-rows').replaceChildren();
  $('#adaptive-usage').replaceChildren();
  if (!result || !selected) {
    $('#adaptive-summary').textContent = 'Для выбранных параметров результатов нет.';
    return;
  }
  const approaches = [
    ['Срез: 1 семестр', 'snapshot'],
    ['Фиксированное: 5 семестров', 'longest'],
    [`Окно по валидации: ${selected.selected_fixed_window} сем.`, 'fixed'],
    ['Адаптивный выбор для каждой истории', 'adaptive'],
  ];
  for (const [label, prefix] of approaches) {
    const tr = document.createElement('tr');
    addTextCell(tr, label + (prefix === 'adaptive' ? ' · исследуемый подход' : ''));
    if (prefix === 'adaptive') tr.className = 'study-row';
    for (const suffix of ['test_ap_mean','test_roc_auc_mean','test_brier_mean'])
      addTextCell(tr, dec(result[`${prefix}_${suffix}`]));
    $('#adaptive-rows').appendChild(tr);
  }
  const delta = result.adaptive_minus_fixed_ap_mean;
  const verdict = delta > 1e-9 ? 'выше, чем у' : delta < -1e-9 ? 'ниже, чем у' : 'такой же, как у';
  $('#adaptive-summary').className = 'result-callout ' + (delta > 1e-9 ? 'positive' : delta < -1e-9 ? 'negative' : 'neutral');
  $('#adaptive-summary').textContent = `В этом сценарии AP адаптивного подхода ${verdict} фиксированного окна, выбранного по валидации: ${Number(result.adaptive_test_ap_mean).toFixed(4)} против ${Number(result.fixed_test_ap_mean).toFixed(4)}. Разность AP: ${delta >= 0 ? '+' : ''}${Number(delta).toFixed(4)}. ${rate === 0 ? 'Пропусков нет, тип пропусков не влияет на контроль.' : 'Результат относится к выбранным условиям пропусков.'}`;
  const config = adaptiveData.design.gate_grid.find(g => g.name === selected.selected_adaptive_gate);
  $('#gate-description').textContent = config ? `Параметры, выбранные на валидации: α = ${config.coverage_weight}, β = ${config.volatility_weight}, γ = ${config.complexity_weight}. Контрольное фиксированное окно: ${selected.selected_fixed_window} сем. При просмотре тестовых метрик параметры не меняются.` : 'Параметры выбранного правила доступны в полном отчёте.';
  const usage = adaptiveData.window_usage.filter(key);
  for (let k = 1; k <= 5; k++) {
    const group = usage.filter(x => x.window === k);
    if (!group.length) continue;
    const fraction = group.reduce((acc,x)=>acc+Number(x.fraction),0)/group.length;
    const row=document.createElement('div'); row.className='barrow';
    const label=document.createElement('span'); label.textContent=k+' сем.';
    const track=document.createElement('div'); track.className='bartrack';
    const fill=document.createElement('div'); fill.className='barfill';
    fill.style.width = Math.max(0,Math.min(100,fraction*100))+'%';
    track.appendChild(fill);
    const value=document.createElement('strong'); value.textContent=percent(fraction);
    row.append(label,track,value); $('#adaptive-usage').appendChild(row);
  }
}
async function loadAdaptive() {
  try {
    const response=await fetch('/api/adaptive-history');
    if (!response.ok) throw new Error('Для v0.4 необходимо запустить python -m edurisk.adaptive');
    adaptiveData=await response.json();
    if (adaptiveData.mode!=='SYNTHETIC_DEMO_ONLY') throw new Error('Допускаются только синтетические данные');
    renderAdaptive();
    renderExperimentOverview();
  } catch(e) { $('#adaptive-summary').textContent = 'Нет данных: '+e.message; $('#outcome-counts').textContent = 'Результаты недоступны'; }
}
for (const name of ['adaptive-model', 'adaptive-pattern', 'adaptive-rate'])
  $('#'+name).addEventListener('change',renderAdaptive);
loadAdaptive();


function renderExperimentOverview() {
  const rows = adaptiveData.comparison;
  const positive = rows.filter(r=>r.adaptive_minus_fixed_ap_mean>1e-9).length;
  const negative = rows.filter(r=>r.adaptive_minus_fixed_ap_mean< -1e-9).length;
  const equal = rows.length-positive-negative;
  const target = $('#outcome-counts'); target.replaceChildren();
  for(const [value,label,cls] of [[positive,'улучшений','positive'],[negative,'ухудшений','negative'],[equal,'без изменений','neutral']]) {
    const item=document.createElement('div'),strong=document.createElement('b'),small=document.createElement('span');
    strong.textContent=value;strong.className=cls;small.textContent=label;item.append(strong,small);target.append(item);
  }
  $('#global-conclusion').textContent='Эффект зависит от условий. Универсального преимущества нет; нужна проверка на независимых временных периодах и когортах.';
  const table=$('#scenario-rows');table.replaceChildren();
  for(const row of rows){
    const tr=document.createElement('tr');
    const delta=Number(row.adaptive_minus_fixed_ap_mean);
    for(const value of [row.model==='random_forest'?'Случайный лес':'Логистическая регрессия',Number(row.missing_rate)===0?'Без пропусков':row.pattern==='mcar_cells'?'Отдельные значения':'Целые семестры',percent(row.missing_rate),Number(row.fixed_test_ap_mean).toFixed(4),Number(row.adaptive_test_ap_mean).toFixed(4)])addTextCell(tr,value);
    addTextCell(tr,(delta>=0?'+':'')+delta.toFixed(4),delta>1e-9?'positive':delta< -1e-9?'negative':'neutral');table.append(tr);
  }
}
$('#export-scenarios').addEventListener('click',()=>{window.location.href='/api/materials/adaptive-table?format=docx';});
