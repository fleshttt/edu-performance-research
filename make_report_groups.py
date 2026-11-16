import json
from pathlib import Path
root=Path(__file__).resolve().parent
labels={'no_activity':'Нет активности','lower_activity':'Низкая активность','higher_activity':'Высокая активность'}
lines=['# Разбор ошибок по курсам и активности','', 'Это исследовательский анализ уже просмотренного теста 2014J. Модели, выбор стратегии и калибровка не настраивались по результатам этого разбора. Порог 0,5 используется для диагностики, а не как готовый порог вмешательства.','']
for profile,title in [('oulad_lr','Логистическая регрессия'),('oulad_rf','Случайный лес')]:
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'reports/runs').glob('*/run.json')]
    record=max((r for r in records if r['profile']==profile and r['status']=='completed'),key=lambda r:r['started_at'])
    result=json.loads((root/'reports/runs'/record['id']/'external_oulad_summary.json').read_text(encoding='utf-8'))
    audit=result['subgroup_audit']
    lines += [f'## {title}', '',f"Запуск: {record['id']}. Граница активности: {audit['activity_boundary']} кликов за {audit['activity_window_days']} дней. Это медиана положительных значений на обучении; тестовые исходы не используются для выбора границы.",'','| Группа | Стратегия | n | Доля риска | AP | Brier | Ложные тревоги | Пропуски |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in audit['rows']:
        vals=[labels.get(row['group'],row['group']),row['method'],row['n'],row['prevalence'],row['ap'],row['brier'],row['false_positives'],row['false_negatives']]
        lines.append('| '+' | '.join('—' if v is None else str(v) for v in vals)+' |')
    rows=[r for r in audit['rows'] if r['dimension']=='activity' and r['method']=='selected' and not r['limited']]
    lines+=['','### Интерпретация','']
    for r in rows:
        lines += [f"{labels[r['group']]}: {r['n']} записей, фактическая доля неуспешных исходов {r['prevalence']:.1%}. При пороге 0,5 — {r['false_positives']} ложных тревог и {r['false_negatives']} пропусков риска; ошибка среди записей группы {r['error']:.1%}. Эти числа описывают разные виды нагрузки: ложные тревоги увеличивают объём ручного рассмотрения, пропуски оставляют часть случаев без сигнала.",'']
lines+=['## Ограничения и следующий шаг','','AP зависит от доли положительного класса: различие AP между курсами само по себе не доказывает разницу качества модели. Малые группы (менее 30 записей или менее 5 исходов одного класса) показываются без метрик. Отсутствие кликов не обязательно означает отсутствие обучения. Повторные записи одного студента зависимы; интервалы для групп здесь не рассчитывались.','','Следующий шаг — отдельный протокол выбора стратегии и калибровки. Сначала определить независимый источник итоговой проверки. На имеющемся OULAD можно отрабатывать разделение обучающих и калибровочных данных, но повторную оценку на 2014J следует явно называть исследовательской. После этого добавить сравнение порогов по валидации: число сигналов, ложные тревоги, пропуски и доступная нагрузка преподавателя.']
(root/'research/21_subgroup_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('research/21_subgroup_results.md')
