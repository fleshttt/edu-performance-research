'use strict';
// Research workspace. Completion is recorded by the researcher, never inferred from dates.
const W = selector => document.querySelector(selector);
const progressKey = 'edurisk.plan.progress.v1';
const statuses = {planned:'Запланировано', active:'В работе', done:'Выполнено'};
let researchPlan, selectedMonth, progress = {}, storageAvailable = true;
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function localMonth() {
  const parts = new Intl.DateTimeFormat('en', {timeZone:'Asia/Tomsk', year:'numeric', month:'2-digit'}).formatToParts(new Date());
  return parts.find(p=>p.type==='year').value+'-'+parts.find(p=>p.type==='month').value;
}
function monthLabel(id) {
  return new Intl.DateTimeFormat('ru',{month:'long',year:'numeric',timeZone:'UTC'}).format(new Date(id+'-01T12:00:00Z'));
}
function progressEntry(id) { return progress[id] || {status:'planned',note:''}; }
function validProgress(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Неверный формат отметок.');
  const clean = {};
  for (const [id, item] of Object.entries(value)) {
    if (!/^20\d{2}-(0[1-9]|1[0-2])$/.test(id) || !item || !Object.hasOwn(statuses,item.status) || typeof item.note !== 'string' || item.note.length > 5000) throw new Error('В файле есть некорректные отметки.');
    clean[id] = {status:item.status,note:item.note};
  }
  return clean;
}
try { progress = validProgress(JSON.parse(localStorage.getItem(progressKey) || '{}')); }
catch { storageAvailable = false; }
function notify(message) { W('#save-message').textContent = message; }
function saveProgress(changes) {
  // Merge individual fields against current storage to preserve work in other tabs.
  const apply = base => {
    const result={...base};
    for(const [id,patch] of Object.entries(changes)) result[id]={status:'planned',note:'',...base[id],...patch};
    return result;
  };
  try {
    const latest=validProgress(JSON.parse(localStorage.getItem(progressKey)||'{}'));
    progress=apply(latest);
    localStorage.setItem(progressKey, JSON.stringify(progress));
    storageAvailable=true;notify('Сохранено в этом браузере.');return true;
  } catch {
    progress=apply(progress);storageAvailable=false;
    notify('Браузер не сохранил изменения. Скачайте резервную копию отметок.');return false;
  }
}
function downloadFile(name, data, type) {
  const url = URL.createObjectURL(new Blob([data],{type}));
  const a = el('a'); a.href=url; a.download=name; document.body.append(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const routeMap = {development:'development',robustness:'experiments',nested:'experiments',uncertainty:'experiments',external:'experiments',overview:'overview',plan:'plan',adaptive:'experiments',ablation:'experiments',models:'experiments',comparison:'experiments',history:'dynamics',examples:'dynamics',cohort:'dynamics',materials:'materials',protocol:'guide',guide:'guide'};
function route() {
  const hash = location.hash.slice(1) || 'development';
  if(hash==='main'){ W('#main').focus();return; }
  const view = routeMap[hash] || 'overview';
  document.querySelectorAll('[data-view]').forEach(node=>node.hidden=node.dataset.view!==view);
  document.querySelectorAll('.sidebar nav a').forEach(a=>{
    const active=routeMap[a.hash.slice(1)]===view; a.classList.toggle('active',active);
    if(active) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current');
  });
  const experiment = ['robustness','nested','uncertainty','external','adaptive','ablation','models','comparison'].includes(hash) ? (hash==='comparison'?'models':hash) : 'adaptive';
  document.querySelectorAll('[data-experiment]').forEach(n=>n.hidden=n.dataset.experiment!==experiment);
  document.querySelectorAll('.experiment-tabs a').forEach(a=>{const active=a.hash==='#'+experiment;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  W('#breadcrumb').textContent = {development:'Разработка',overview:'Рабочий стол',plan:'План диссертации',experiments:'Эксперименты',dynamics:'Динамика студента',materials:'Материалы исследования',guide:'Как работать'}[view];
  if (view==='dynamics' && ['examples','cohort'].includes(hash)) W('#'+hash).scrollIntoView();
  else window.scrollTo({top:0,behavior:'instant'});
}
window.addEventListener('hashchange',route);
route();
function materialLink(id) {
  const m=researchPlan.materials.find(x=>x.id===id); if(!m)return null;
  const link=el('a',m.title,'material-link');link.href=m.url+'?format=docx';link.setAttribute('download','');return link;
}
function renderOverview() {
  const now=localMonth();
  const current=researchPlan.months.find(m=>m.id===now);
  const first=researchPlan.months[0], last=researchPlan.months.at(-1);
  const target=current || (now < first.id ? first : last);
  const stage=researchPlan.stages.find(s=>s.id===target.stage_id);
  W('#current-period').textContent=current?monthLabel(now):'План: '+monthLabel(first.id)+' — '+monthLabel(last.id);
  W('#current-stage').textContent=current?'Этап '+stage.id+' · '+stage.title:'Период плана';
  W('#next-work').textContent=target.tasks.join(' ');
  W('#next-deliverable').textContent=target.deliverables.join(' ');
  W('#open-current').onclick=()=>{selectedMonth=target.id;W('#plan-year').value=String(target.year);renderPlan();location.hash='plan';};
  const done=researchPlan.months.filter(m=>progressEntry(m.id).status==='done').length;
  W('#plan-progress').textContent=done+' / '+researchPlan.months.length;
  W('#progress-caption').textContent='месяцев отмечено выполненными';
  W('#overview-progress').value=done;W('#overview-progress').max=researchPlan.months.length;
  W('#researcher').textContent=researchPlan.researcher;
  W('#supervisor').textContent=researchPlan.supervisor;
}
function renderPlan() {
  const year=Number(W('#plan-year').value);
  const months=researchPlan.months.filter(m=>m.year===year);
  if(!months.some(m=>m.id===selectedMonth)) selectedMonth=months[0].id;
  const list=W('#month-list');list.replaceChildren();
  months.forEach(m=>{
    const entry=progressEntry(m.id), button=el('button',undefined,'month-button'+(m.id===selectedMonth?' selected':''));
    button.type='button';button.setAttribute('aria-pressed',String(m.id===selectedMonth));
    button.append(el('span',monthLabel(m.id)),el('small',m.id===localMonth()?'Текущий месяц · '+statuses[entry.status]:statuses[entry.status],'status-'+entry.status));
    button.onclick=()=>{selectedMonth=m.id;renderPlan();};list.append(button);
  });
  const m=researchPlan.months.find(m=>m.id===selectedMonth), stage=researchPlan.stages.find(s=>s.id===m.stage_id),entry=progressEntry(m.id);
  W('#month-title').textContent=monthLabel(m.id);W('#month-stage').textContent='Этап '+stage.id+' · '+stage.title;
  W('#month-tasks').replaceChildren(...m.tasks.map(t=>el('li',t)));
  W('#month-deliverables').replaceChildren(...m.deliverables.map(t=>el('li',t)));
  W('#month-status').value=entry.status;W('#month-note').value=entry.note;
  W('#month-materials').replaceChildren();
  const links=(m.material_ids||[]).map(materialLink).filter(Boolean);
  W('#month-materials').append(...(links.length?links:[el('p','Для этого месяца готовых материалов в проекте пока нет.','muted')]));
  W('#stage-deliverables').replaceChildren(...stage.deliverables.map(t=>el('li',t)));
  notify(storageAvailable?'Отметки сохраняются только в этом браузере.':'Не удалось прочитать хранилище. Используйте экспорт отметок.');
}
function renderMaterials() {
  const query=W('#material-search').value.toLocaleLowerCase('ru').trim();
  const list=W('#material-list');list.replaceChildren();
  const items=researchPlan.materials.filter(m=>(m.title+' '+m.description+' '+m.category).toLocaleLowerCase('ru').includes(query));
  W('#material-count').textContent='Найдено материалов: '+items.length;
  for(const m of items){
    const card=el('article',undefined,'material-card');
    card.append(el('span',m.category,'eyebrow'),el('h3',m.title),el('p',m.description));
    const a=el('a','Скачать Word ↓','text-link');a.href=m.url+'?format=docx';a.setAttribute('download','');card.append(a);list.append(card);
  }
  if(!items.length)list.append(el('p','Ничего не найдено. Попробуйте другой запрос.','empty-state'));
}
async function loadPlan(){
  try {
    const response=await fetch('/api/research-plan');if(!response.ok)throw new Error('Сервер не вернул план.');researchPlan=await response.json();
    selectedMonth=researchPlan.months.some(m=>m.id===localMonth())?localMonth():researchPlan.months[0].id;
    W('#plan-year').value=String(researchPlan.months.find(m=>m.id===selectedMonth).year);
    W('#plan-error').hidden=true;W('#plan-content').hidden=false;
    renderOverview();renderPlan();renderMaterials();
    W('#plan-sources').replaceChildren(...researchPlan.sources.map(s=>el('li',s.title+' · '+s.filename)));
    W('#plan-notes').replaceChildren(...researchPlan.notes.map(s=>el('p',s)));
    W('#alignment-notes').replaceChildren();
    for(const n of researchPlan.alignment_notes){const d=el('details');d.append(el('summary',n.title),el('p',n.detail),el('p',n.action));W('#alignment-notes').append(d);}
    W('#requirements').replaceChildren(...researchPlan.requirements.map(r=>{const item=el('li');item.append(el('strong',r.title),el('span',r.due_label,'muted'));return item;}));
    W('#stage-timeline').replaceChildren(...researchPlan.stages.map(s=>{const b=el('button',undefined,'stage-card');b.type='button';b.append(el('small','Этап '+s.id+' · '+s.start+' — '+s.end),el('strong',s.title));b.onclick=()=>{selectedMonth=s.start;W('#plan-year').value=String(s.year);renderPlan();location.hash='plan';};return b;}));
  }catch(e){W('#plan-error').hidden=false;W('#plan-error-text').textContent=e.message;W('#next-work').textContent='План пока не загружен. Повторите загрузку в разделе «План диссертации».';}
}
W('#retry-plan').onclick=loadPlan;
W('#plan-year').onchange=renderPlan;
W('#month-status').onchange=()=>{const saved=saveProgress({[selectedMonth]:{status:W('#month-status').value}});renderOverview();renderPlan();if(!saved)notify('Изменение пока только в памяти. Скачайте резервную копию отметок.');};
W('#month-note').oninput=()=>saveProgress({[selectedMonth]:{note:W('#month-note').value}});
W('#material-search').oninput=()=>{if(researchPlan)renderMaterials();};
W('#export-progress').onclick=()=>downloadFile('edurisk-progress.json',JSON.stringify({schema:'edurisk-progress-v1',exported_at:new Date().toISOString(),months:progress},null,2),'application/json');
W('#import-progress').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  try{
    if(file.size>500000)throw new Error('Файл слишком большой.');
    const value=JSON.parse(await file.text());if(value.schema!=='edurisk-progress-v1')throw new Error('Выберите резервную копию EduRisk.');
    const incoming=validProgress(value.months);
    if(Object.keys(incoming).some(id=>!researchPlan.months.some(m=>m.id===id)))throw new Error('В файле есть месяцы вне плана.');
    // Fill only empty records: importing a backup never overwrites existing personal work.
    try { progress=validProgress(JSON.parse(localStorage.getItem(progressKey)||'{}')); } catch {}
    const changes={};for(const [id,entry] of Object.entries(incoming)){if(!progress[id] || (progress[id].status==='planned' && !progress[id].note))changes[id]=entry;}
    const saved=saveProgress(changes);renderOverview();renderPlan();notify(saved?'Восстановлено месяцев: '+Object.keys(changes).length+'. Существующие отметки сохранены.':'Копия прочитана, но браузер не сохранил изменения. Скачайте резервную копию перед закрытием.');
  }catch(error){notify('Не удалось импортировать: '+error.message);}finally{e.target.value='';}
};
loadPlan();

W('#export-plan-word').onclick=async()=>{
  try{const r=await fetch('/api/word/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({months:progress})});if(!r.ok)throw new Error('Не удалось создать Word');downloadFile('План диссертации.docx',await r.blob(),'application/vnd.openxmlformats-officedocument.wordprocessingml.document');}catch(e){notify(e.message);}
};
