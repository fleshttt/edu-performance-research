"""Compare archived shared and separate calibration runs without tuning models."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
names={'oulad_lr':'Логистическая регрессия','oulad_rf':'Случайный лес','oulad_hist_gradient_boosting':'Градиентный бустинг','oulad_extra_trees':'Extra Trees','oulad_knn':'kNN','oulad_gaussian_nb':'Наивный Байес'}
def build():
 records={}
 for p in (ROOT/'reports/runs').glob('*/run.json'):
  meta=json.loads(p.read_text(encoding='utf-8'))
  if meta['status']!='completed':continue
  key=meta['profile']
  if key not in records or meta['started_at']>records[key][0]['started_at']:
   records[key]=(meta,json.loads((p.parent/'external_oulad_summary.json').read_text(encoding='utf-8')))
 lines=['# Результаты отдельной калибровки','','Проверено разделение студентов 2014B между выбором стратегии и калибровкой. Это исследовательское сравнение на уже просмотренном тесте 2014J. Настройки не подбирались по его результатам. Меньший объём данных для выбора стратегии также влияет на итог, поэтому сравнение не выделяет причинный эффект только калибровки.','','| Модель | Brier старый после | Brier новый до | Brier новый после | ECE новый до | ECE новый после |','|---|---:|---:|---:|---:|---:|']
 details=[]
 for profile,label in names.items():
  oldmeta,old=records[profile];meta,new=records[profile+'_separate']
  if old['source_file_hashes']!=new['source_file_hashes'] or old['n_rows']!=new['n_rows']:raise ValueError('Input data mismatch')
  audit=new['calibration_audit']
  assert audit['overlap_students']==0
  before=new['metrics'][new['selection']['selected_method']];after=new['metrics']['selected_calibrated'];prior=old['metrics']['selected_calibrated']
  lines.append(f"| {label} | {prior['brier']:.5f} | {before['brier']:.5f} | {after['brier']:.5f} | {before['ece10']:.5f} | {after['ece10']:.5f} |")
  direction='ухудшилась' if after['brier']>before['brier'] else 'улучшилась' if after['brier']<before['brier'] else 'не изменилась'
  details += [f'## {label}','',f"Выбор стратегии: {audit['selection_rows']} записей, {audit['selection_students']} студентов. Калибровка: {audit['calibration_rows']} записей, {audit['calibration_students']} студентов. Пересечение: 0.",'',f"Выбрана стратегия {new['selection']['selected_method']}, окно сравнения {new['selection']['fixed_window_days']} дней. В исходном протоколе: {old['selection']['selected_method']}, {old['selection']['fixed_window_days']} дней.",'',f"После калибровки ошибка вероятностей по Brier {direction}: {before['brier']:.5f} → {after['brier']:.5f}. Меньшее значение лучше. ECE изменился с {before['ece10']:.5f} до {after['ece10']:.5f}; он зависит от группировки вероятностей. AP до калибровки {before['ap']:.5f}, после {after['ap']:.5f}. AP описывает порядок по риску, а не долю верных решений.",'',f"Исходный запуск: {oldmeta['id']}. Новый запуск: {meta['id']}.",'']
 lines+=['']+details+['## Что выполнено и что дальше','','Технически устранено пересечение студентов между выбором стратегии и калибровкой. Это не гарантирует улучшение метрик и не устраняет перенос между периодами. Сохраняются ограничения временной доступности оценок OULAD и применимости к другим университетам.','','Следующая задача разработки — оценка порогов сигнала с указанием ложных тревог, пропусков и нагрузки преподавателя. Для обоснованного окончательного выбора нужна новая независимая проверка. Отметки индивидуального плана, публикации и аттестация этим экспериментом не закрываются автоматически.']
 out=ROOT/'research/24_separate_calibration_results.md';out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out)
if __name__=='__main__':build()
