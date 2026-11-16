


// v0.5: read-only synthetic forward, disjoint-cohort and structured missingness comparisons.
let robustnessData = null;
function renderRobustness() {
  if (!robustnessData) return;
  const protocol=$('#robustness-protocol').value;
  const family=$('#robustness-model').value;
  const mechanism=$('#robustness-mechanism').value;
  const matching=x=>x.protocol===protocol && x.model===family && x.mechanism===mechanism;
  const rows=robustnessData.aggregate_results.filter(matching);
  const diffs=robustnessData.comparative_deltas.filter(matching);
  $('#robustness-rows').replaceChildren();
  $('#robustness-bars').replaceChildren();
  if (!rows.length) {
    $('#robustness-summary').textContent='Для выбранного сценария результатов нет.';
    return;
  }
  const fixed=rows.find(x=>x.method==='fixed_val');
  const methods=[
    ['Срез 1 семестр','snapshot_1'],
    ['Самое длинное окно','longest'],
    ['Фиксированное окно (по валидации)','fixed_val'],
    ['Адаптивный алгоритм v0.4','adaptive_gate'],
    ['Равномерное усреднение','uniform_ensemble'],
    ['Взвешенное усреднение','weighted_ensemble'],
  ];
  for (const [label,method] of methods) {
    const record=rows.find(x=>x.method===method);
    if (!record) continue;
    const tr=document.createElement('tr');
    addTextCell(tr,label,method==='adaptive_gate'?'best':'');
    for (const k of ['test_average_precision_mean','test_roc_auc_mean','test_brier_mean']) addTextCell(tr,dec(record[k]));
    const delta=(Number(record.test_average_precision_mean)-Number(fixed.test_average_precision_mean));
    addTextCell(tr,(delta>=0?'+':'')+dec(delta));
    $('#robustness-rows').appendChild(tr);
  }
  $('#robustness-summary').textContent=`Протокол: ${protocol}. Разбиений: ${fixed.n_folds}. Доля пропусков задана 25% для ненулевых сценариев (реализованная доля может отличаться). Результаты на синтетических данных.`;
  for (const [label,method] of methods.filter(([,m])=>['adaptive_gate','uniform_ensemble','weighted_ensemble'].includes(m))) {
    const d=diffs.find(x=>x.method===method);
    if(!d) continue;
    const row=document.createElement('div');row.className='barrow';
    const t=document.createElement('span');t.textContent=label;
    const track=document.createElement('div');track.className='bartrack';
    const fill=document.createElement('div');fill.className='barfill';
    // Visualize magnitude only, signed value remains explicit in the label.
    fill.style.width=Math.min(100,Math.abs(d.mean_delta_ap)*1000)+'%';
    track.append(fill);
    const val=document.createElement('strong');val.textContent=(d.mean_delta_ap>=0?'+':'')+dec(d.mean_delta_ap);
    row.append(t,track,val);$('#robustness-bars').append(row);
  }
}
async function loadRobustness() {
  try {
    const response=await fetch('/api/robustness');
    if(!response.ok) throw new Error('Для v0.5 запустите python -m edurisk.robustness');
    robustnessData=await response.json();
    if(robustnessData.mode!=='SYNTHETIC_DEMO_ONLY') throw new Error('Ожидались только синтетические данные');
    renderRobustness();
  } catch(e) {$('#robustness-summary').textContent='Нет результатов: '+e.message;}
}
for (const name of ['robustness-protocol','robustness-model','robustness-mechanism'])
  $('#'+name).addEventListener('change',renderRobustness);
loadRobustness();


// v0.6: repeated synthetic cohorts, nested rolling validation, test-only evaluation.
let nestedData = null;
const nestedLabels={
  snapshot_1:'Один семестр',longest:'Максимальное окно',
  fixed_val:'Фиксированное окно',adaptive_gate:'Адаптивный выбор',
  uniform_ensemble:'Равномерный ансамбль',weighted_ensemble:'Взвешенный ансамбль',
  selected_nested:'Стратегия по вложенной проверке',
};
function renderNested() {
  if(!nestedData) return;
  const family=$('#nested-family').value;
  const mechanism=$('#nested-mechanism').value;
  const rows=nestedData.summaries.filter(r=>r.family===family && r.mechanism===mechanism);
  $('#nested-rows').replaceChildren();$('#nested-choices').replaceChildren();
  if(!rows.length){$('#nested-summary').textContent='Для выбранного сочетания результатов нет.';return;}
  const selected=rows.find(r=>r.method==='selected_nested');
  for(const method of ['fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble','selected_nested']) {
    const record=rows.find(x=>x.method===method);
    if(!record) continue;
    const tr=document.createElement('tr');
    addTextCell(tr,nestedLabels[method],method==='selected_nested'?'best':'');
    for(const key of ['test_ap_mean','test_roc_auc_mean','test_brier_mean']) addTextCell(tr,dec(record[key]));
    const delta=record.delta_ap_vs_fixed;
    addTextCell(tr,delta==null?'—':(delta>=0?'+':'')+dec(delta));
    addTextCell(tr,record.delta_ap_ci_low == null ? '—' : '['+dec(record.delta_ap_ci_low)+'; '+dec(record.delta_ap_ci_high)+']');
    $('#nested-rows').appendChild(tr);
  }
  const counts=nestedData.selection_counts.filter(c=>c.mechanism===mechanism && c.family===family);
  const n=counts.reduce((sum,x)=>sum+x.n_selected,0);
  for(const method of ['fixed_val','adaptive_gate','uniform_ensemble','weighted_ensemble']) {
    const count=counts.find(x=>x.chosen_method===method)?.n_selected || 0;
    const row=document.createElement('div');row.className='barrow';
    const label=document.createElement('span');label.textContent=nestedLabels[method];
    const track=document.createElement('div');track.className='bartrack';
    const fill=document.createElement('div');fill.className='barfill';
    fill.style.width=(n?100*count/n:0)+'%';track.appendChild(fill);
    const value=document.createElement('strong');value.textContent=count+'/'+n;
    row.append(label,track,value);$('#nested-choices').append(row);
  }
  const delta=selected.delta_ap_vs_fixed;
  $('#nested-summary').textContent=`Повторов: ${selected.n_replicates}; средняя доля положительных меток: ${percent(selected.mean_prevalence)}. Выбор метода: только по внутренним проверкам. Разница AP с фиксированным окном: ${delta>=0?'+':''}${dec(delta)}. Интервалы отражают изменчивость синтетических генераций.`;
}
async function loadNested() {
  try {
    const response=await fetch('/api/nested-selection');
    if(!response.ok) throw new Error('Запустите python -m edurisk.nested_selection');
    nestedData=await response.json();
    if(nestedData.mode!=='SYNTHETIC_DEMO_ONLY') throw new Error('Требуются синтетические данные');
    renderNested();
  } catch(e){$('#nested-summary').textContent='Нет данных: '+e.message;}
}
for(const name of ['nested-family','nested-mechanism'])
  $('#'+name).addEventListener('change',renderNested);
loadNested();

// v0.7: validation-fold performance uncertainty; synthetic aggregates only.
let uncData=null;
const uncLabels={
  selected_nested:'v0.6 · максимальная AP по OOF',
  selected_fold_mean:'Средняя AP по семестрам',
  selected_stability_1:'Средняя AP − 1 SE',
  selected_stability_2:'Средняя AP − 2 SE',
  selected_one_se:'Правило одной SE',
};
function renderUncertainty() {
  if(!uncData)return;
  const family=$('#unc-family').value, mechanism=$('#unc-mechanism').value;
  const rows=uncData.summaries.filter(r=>r.family===family&&r.mechanism===mechanism);
  $('#unc-rows').replaceChildren();
  if(!rows.length){$('#unc-summary').textContent='Результаты не найдены';return;}
  for(const method of Object.keys(uncLabels)){
    const item=rows.find(r=>r.policy===method);if(!item)continue;
    const tr=document.createElement('tr');
    addTextCell(tr,uncLabels[method]);
    addTextCell(tr,dec(item.ap_mean));
    addTextCell(tr,dec(item.brier_mean));
    addTextCell(tr,(item.delta_v06_mean >= 0 ? '+':'')+dec(item.delta_v06_mean));
    addTextCell(tr,'['+dec(item.delta_v06_ci_lo)+'; '+dec(item.delta_v06_ci_hi)+']');
    addTextCell(tr,String(item.positive_vs_v06)+' из '+item.replicates);
    $('#unc-rows').append(tr);
  }
  const chosen=rows.find(r=>r.policy==='selected_stability_1');
  $('#unc-summary').textContent=`Повторов: ${chosen.replicates}; тест — семестр 9; окна 1, 3 и 5. ΔAP для штрафа 1 SE: ${chosen.delta_v06_mean>=0?'+':''}${dec(chosen.delta_v06_mean)}. Интервалы — только описательные для искусственных генераций.`;
}
async function loadUncertainty(){
  try {
    const response=await fetch('/api/uncertainty-selection');
    if(!response.ok)throw new Error('Отчёт v0.7 ещё не рассчитан');
    uncData=await response.json();
    if(uncData.mode!=='SYNTHETIC_DEMO_ONLY')throw new Error('Только синтетические исследования');
    renderUncertainty();
  }catch(err){$('#unc-summary').textContent='Нет данных: '+err.message;}
}
for(const id of ['unc-family','unc-mechanism'])$('#'+id).addEventListener('change',renderUncertainty);
loadUncertainty();


// v0.8: an aggregated-only, clearly marked OULAD external validation view.
let externalRequest = 0;
const externalNames = {fixed_val:'Фиксированное окно по валидации',adaptive_gate:'Адаптивное правило',uniform_ensemble:'Равномерный ансамбль',weighted_ensemble:'Взвешенный ансамбль',selected_calibrated:'Выбранная стратегия после калибровки'};
function renderExternalConclusions(result, fixture) {
  const host=$('#external-conclusions');
  const selected=result.selection.selected_method;
  const fixed=result.metrics.fixed_val, chosen=result.metrics[selected], calibrated=result.metrics.selected_calibrated;
  const change=(a,b)=>(a>b?'вырос':a<b?'снизился':'не изменился');
  const num=x=>Number(x).toFixed(5);
  const sections=[
    ['Что именно проверяли', fixture
      ? 'Это искусственный набор для проверки работы программы. Его цифры не позволяют судить о качестве прогноза на реальных студентах.'
      : `По информации, доступной к 84-му дню курса, модель оценивает риск итогового результата Fail или Withdrawn. В анализ входят только ещё зарегистрированные на этот день студенты. Тест содержит ${result.n_rows.test} записей о прохождении курсов; один студент может иметь несколько записей. Речь идёт о риске неуспешного завершения курса, а не о диагнозе или неизбежном отчислении.`],
    ['Как читать показатели', 'AP показывает качество списка студентов, отсортированного по риску: выше — лучше. AP = 0,73 не означает 73% верных прогнозов. ROC-AUC характеризует разделение двух классов, выше — лучше. Brier оценивает ошибку вероятностного прогноза, ниже — лучше. ECE10 сравнивает прогнозируемый риск с фактической частотой исходов в десяти группах; ниже — лучше, но результат зависит от группировки. Хорошее ранжирование не гарантирует достоверных вероятностей.'],
    ['Что получилось у выбранной стратегии', `На валидации выбрана стратегия «${externalNames[selected]}». По сравнению с фиксированным окном её AP ${change(chosen.ap,fixed.ap)}: ${num(fixed.ap)} → ${num(chosen.ap)}. Разница ${num(chosen.ap-fixed.ap)} относится к шкале AP от 0 до 1; её нельзя переводить в число предотвращённых неуспехов. Объединение окон использует несколько периодов наблюдения вместо одного. Сравнение относится к этому протоколу и этой тестовой когорте.`],
    ['Сработала ли адаптация', `У адаптивного правила AP = ${num(result.metrics.adaptive_gate.ap)}, у фиксированного окна — ${num(fixed.ap)}. ${result.metrics.adaptive_gate.ap<fixed.ap?'В этом запуске адаптивное правило уступило базовому подходу. Поэтому преимущество ансамбля нельзя выдавать за подтверждение эффективности адаптивного выбора окна.':'Направление сравнения видно по AP; одного запуска недостаточно для общего вывода об адаптации.'} Следующий шаг — разобрать ошибки по курсам и активности, сформулировать проверяемую гипотезу и проверить её на новых данных.`],
    ['Можно ли доверять числу риска', `После калибровки Brier ${change(calibrated.brier,chosen.brier)}: ${num(chosen.brier)} → ${num(calibrated.brier)}; ECE10 ${change(calibrated.ece10,chosen.ece10)}: ${num(chosen.ece10)} → ${num(calibrated.ece10)}. Для этих метрик рост означает ухудшение. Калибровка пытается согласовать вероятности с частотой исходов, но её полезность нужно проверять отдельно. Причина изменения качества здесь не установлена. Для её проверки нужно учитывать перенос между периодами и протокол разделения данных, указанный над таблицей.`],
    ['Насколько устойчив вывод', `Интервал рядом с таблицей получен ${result.bootstrap.resamples} парными повторными выборками по студентам. Он описывает неопределённость разницы AP внутри этой тестовой когорты при уже обученных моделях. Он не учитывает все варианты обучения и не доказывает переносимость на другой университет. Просмотренный тест 2014J уже нельзя использовать для настройки и затем называть независимой проверкой.`],
    ['Как использовать результат в разработке', 'Сохранить текущие модели и отчёты как точку сравнения. Далее разделить выбор модели и калибровку, заранее описать проверку на новых периодах или независимом наборе, оценить ошибки по курсам и подобрать порог с учётом допустимого числа ложных тревог. До такой проверки результат подходит для исследовательского прототипа. Эффект поддержки студентов и снижение числа неуспехов здесь не измерялись.'],
    ['Что означает отказ от ответа ниже', 'Покрытие — доля записей, для которых модель оставила ответ после отсечения вероятностей около 0,5. Ошибка рассчитана только среди оставшихся записей. Меньшее покрытие означает больше случаев для ручного рассмотрения, а не автоматическое улучшение системы. Отбрасывание неуверенных ответов само по себе не доказывает надёжности оставшихся.']
  ];
  host.replaceChildren();
  for(const [title,copy] of sections){const section=document.createElement('section');const h=document.createElement('h3');h.textContent=title;const p=document.createElement('p');p.textContent=copy;section.append(h,p);host.append(section);}
}
async function loadExternal() {
  const token=++externalRequest;
  const status=$('#external-status'),body=$('#external-rows'),details=$('#external-details');
  body.replaceChildren();$('#external-selective').replaceChildren();$('#external-downloads').replaceChildren();
  $('#external-conclusions').replaceChildren();$('#calibration-audit').textContent='';
  $('#external-groups').replaceChildren();$('#external-groups-note').textContent='';
  details.textContent='';$('#external-comparison').textContent='';$('#external-provenance').textContent='';
  status.textContent='Загружаем выбранный отчёт…';
  try {
    const query=new URLSearchParams({dataset:$('#external-dataset').value,family:$('#external-family').value,protocol:$('#external-calibration').value});
    const response=await fetch('/api/external-validation?'+query);
    if (!response.ok) throw new Error('Отчёт недоступен');
    const result=await response.json();if(token!==externalRequest)return;
    if(result.mode==='NOT_RUN'){status.textContent='Для выбранного источника и семейства нет завершённого запуска. Откройте раздел «Разработка».';return;}
    const fixture=result.mode==='SYNTHETIC_SCHEMA_FIXTURE';
    renderExternalConclusions(result,fixture);
    const ca=result.calibration_audit;
    $('#calibration-audit').textContent=result.design.calibration_protocol==='separate'?`Отдельная калибровка: выбор стратегии — ${ca.selection_rows} записей (${ca.selection_students} студентов); калибровка — ${ca.calibration_rows} записей (${ca.calibration_students} студентов). Пересечение студентов: ${ca.overlap_students}. Разделение фиксировано без использования исходов. Это исследовательское сравнение на ранее просмотренном тесте. Меньшая выборка выбора стратегии тоже влияет на результат — разницу нельзя приписать только калибровке.`:'Исходный протокол: выбор стратегии и калибровка используют одну валидацию 2014B.';

    const audit=result.subgroup_audit;
    $('#external-groups-note').textContent=audit?`Исследовательский разбор уже просмотренного теста. Активность — число кликов за ${audit.activity_window_days} дней; граница низкой и высокой активности ${audit.activity_boundary}, рассчитана только на обучении. Порог риска 0,5 задан для диагностики и не оптимизирован. Ложная тревога — сигнал риска при успешном исходе; пропуск — отсутствие сигнала при неуспешном. Для групп менее 30 записей или менее 5 исходов одного класса метрики скрыты (—). AP зависит от доли риска: напрямую ранжировать курсы по AP некорректно.`:'В этом сохранённом запуске разбора групп ещё нет. Выполните новый запуск в разделе «Разработка».';
    const groupLabels={no_activity:'Нет активности',lower_activity:'Низкая активность',higher_activity:'Высокая активность'};
    const methodLabels={fixed_val:'Фиксированное окно',selected:'Выбранная стратегия',selected_calibrated:'После калибровки'};
    for(const row of audit?.rows||[]){const tr=document.createElement('tr');for(const value of [groupLabels[row.group]||'Курс '+row.group,methodLabels[row.method],row.n,row.prevalence==null?'—':percent(row.prevalence),row.ap==null?'—':row.ap.toFixed(5),row.brier==null?'—':row.brier.toFixed(5),row.false_positives??'—',row.false_negatives??'—'])addTextCell(tr,value);$('#external-groups').append(tr);}
    status.textContent=fixture?'ИСКУССТВЕННЫЙ НАБОР — техническая проверка, не результаты на оригинальном OULAD.':'ОРИГИНАЛЬНЫЙ OULAD · отложенная проверка 2014J';
    for(const key of Object.keys(externalNames)){
      const m=result.metrics[key];if(!m)continue;
      const tr=document.createElement('tr');addTextCell(tr,externalNames[key]);
      for(const metric of ['ap','roc_auc','brier','ece10'])addTextCell(tr,Number(m[metric]).toFixed(5));body.append(tr);
    }
    const n=result.n_rows,selected=result.selection.selected_method;
    details.textContent=`Записей: обучение ${n.train}, валидация ${n.val}, тест ${n.test}. Доля положительного класса на тесте: ${percent(result.classes.test)}. Выбрана на валидации: ${externalNames[selected]||selected}; контрольное окно ${result.selection.fixed_window_days} дней. Семейство в переключателе выбирается для просмотра, а не как победитель по тесту.`;
    const before=result.metrics[selected],after=result.metrics.selected_calibrated,delta=result.bootstrap.delta_ap;
    $('#external-comparison').textContent=`Разность AP выбранной стратегии и фиксированного окна до калибровки: ${delta>=0?'+':''}${delta.toFixed(5)}. 95% интервал: ${result.bootstrap.ci95?result.bootstrap.ci95.join(' … '):'не рассчитан'} (${result.bootstrap.resamples} повторов по студентам). Brier после калибровки ${after.brier>before.brier?'ухудшился':after.brier<before.brier?'улучшился':'не изменился'}: ${before.brier.toFixed(5)} → ${after.brier.toFixed(5)}. Интервал условен на одной отложенной когорте.`;
    const prov=result.provenance;
    $('#external-provenance').textContent=prov?.kind==='uci_download'?`Источник: UCI, CC BY 4.0. Семь файлов совпадают с квитанцией загрузки. SHA-256 архива: ${prov.archive_sha256}. Независимая цифровая подпись не проверялась.`:fixture?'Источник: искусственный генератор.':'Источник заявлен пользователем; подлинность автоматически не подтверждена.';
    if(result.run_id){for(const [key,label] of [['summary','Полный отчёт Word'],['table','Метрики Word'],['log','Журнал Word']]){const link=document.createElement('a');link.className='material-link';link.href='/api/runs/'+result.run_id+'/'+key+'?format=docx';link.download='';link.textContent=label;$('#external-downloads').append(link);}}
    for(const r of result.selective){const tr=document.createElement('tr');for(const x of [r.margin,percent(r.coverage),r.n,percent(r.classification_error)])addTextCell(tr,x);$('#external-selective').append(tr);}
  }catch(error){if(token===externalRequest)status.textContent='Ошибка: '+error.message;}
}
for(const id of ['external-dataset','external-family','external-calibration'])$('#'+id).addEventListener('change',loadExternal);
$('#external-refresh').addEventListener('click',loadExternal);
loadExternal();
