'use strict';
const D = selector => document.querySelector(selector);
let runPoll;
const runNames = {oulad_lr:'OULAD · логистическая регрессия',oulad_rf:'OULAD · случайный лес',fixture_lr:'Логистическая регрессия',fixture_rf:'Случайный лес'};
const runStates = {running:'Выполняется',completed:'Завершён',failed:'Ошибка',interrupted:'Прерван перезапуском сервера'};
function dNode(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
async function refreshDevelopment(){
  clearTimeout(runPoll);
  try{
    const response=await fetch('/api/development');if(!response.ok)throw new Error('Сервер не вернул состояние проекта');
    const data=await response.json();
    D('#data-files').replaceChildren(...data.files.map(file=>{
      const row=dNode('li');row.append(dNode('code',file.name),dNode('span',!file.exists?'Отсутствует':file.error|| (file.missing_columns.length?'Нет колонок: '+file.missing_columns.join(', '):'Структура проверена'),file.exists&&!file.error&&!file.missing_columns.length?'positive':'muted'));return row;
    }));
    D('#data-state').textContent=data.schema_ready?'Семь CSV готовы. Источник и совпадение контрольных сумм фиксируются в отчёте каждого запуска.':'Оригинальные данные ещё не готовы к запуску. Технический тест использует отдельный искусственный набор.';
    D('#run-list').replaceChildren();
    for(const r of data.runs){
      const row=dNode('article',undefined,'run-card');
      row.append(dNode('h3',runNames[r.profile]||r.profile),dNode('p',new Date(r.started_at).toLocaleString('ru-RU')+' · '+runStates[r.status]));
      row.append(dNode('small',(r.mode==='REAL_OULAD_EXTERNAL'?'Внешняя проверка OULAD · ':'Искусственный набор · ')+r.id.slice(0,8)));
      if(r.error)row.append(dNode('p',r.error,'negative'));
      if(r.metrics)row.append(dNode('p',`После калибровки: AP ${r.metrics.ap} · Brier ${r.metrics.brier} · ROC-AUC ${r.metrics.roc_auc}`));
      const links=dNode('div',undefined,'actions');
      for(const [artifact,label] of (r.status==='completed'?[['summary','Отчёт Word'],['table','Метрики Word'],['log','Журнал Word']]:r.status==='failed'?[['log','Журнал Word']]:[])){
        const a=dNode('a',label,'material-link');a.href='/api/runs/'+r.id+'/'+artifact+'?format=docx';a.download='';links.append(a);
      }
      row.append(links);D('#run-list').append(row);
    }
    if(!data.runs.length)D('#run-list').append(dNode('p','Новых запусков пока нет. Сохранённые эксперименты v0.4–v0.8 доступны в разделе «Эксперименты».','muted'));
    for(const opt of D('#run-profile').options) if(opt.value.startsWith('oulad_')) opt.disabled=!data.schema_ready;
    const active=data.runs.some(r=>r.status==='running');D('#start-run').disabled=active;
    D('#run-message').textContent=active?'Эксперимент выполняется. Можно перейти в другой раздел; результат появится в журнале.':'Готово к запуску. Каждый результат сохраняется отдельно от исходных отчётов.';
    runPoll=setTimeout(refreshDevelopment,active?2000:15000);
  }catch(error){D('#run-message').textContent='Ошибка: '+error.message;D('#start-run').disabled=false;runPoll=setTimeout(refreshDevelopment,15000);}
}
D('#start-run').onclick=async()=>{
  D('#start-run').disabled=true;D('#run-message').textContent='Запускаем эксперимент…';
  try{
    const response=await fetch('/api/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({profile:D('#run-profile').value+(D('#run-calibration').value==='separate'?'_separate':'')})});
    const result=await response.json();if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'Не удалось запустить эксперимент');
    await refreshDevelopment();
  }catch(error){D('#run-message').textContent=error.message;D('#start-run').disabled=false;}
};
D('#refresh-development').onclick=refreshDevelopment;
refreshDevelopment();

Object.assign(runNames,{"oulad_hist_gradient_boosting": "Градиентный бустинг", "oulad_extra_trees": "Extra Trees", "oulad_knn": "Ближайшие соседи (kNN)", "oulad_gaussian_nb": "Наивный Байес", "fixture_hist_gradient_boosting": "Градиентный бустинг", "fixture_extra_trees": "Extra Trees", "fixture_knn": "Ближайшие соседи (kNN)", "fixture_gaussian_nb": "Наивный Байес"});

for(const [key,value] of Object.entries({...runNames}))runNames[key+"_separate"]=value+" · отдельная калибровка";
