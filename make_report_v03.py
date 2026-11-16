"""Rebuild Russian v0.3 research report and two plots from saved synthetic CSVs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
r = ROOT / 'reports'
a = pd.read_csv(r / 'ablation_aggregates_synthetic.csv')
s = pd.read_csv(r / 'ablation_selections_synthetic.csv')

fmt = lambda x: f'{x:.3f}'
lines = [
    '# Отчёт об эксперименте v0.3 (синтетическая демонстрация)',
    '',
    '**Тема:** влияние длины истории успеваемости и пропущенных измерений на качество прогноза.',
    '**Источник данных:** искусственно сгенерированные записи, 420 студентов, 10 смоделированных семестров.',
    '',
    '**Значение для диссертации:** этап проверки методики и отбора направлений исследования,',
    'не доказательство научной новизны и не оценка работы на настоящих образовательных данных.',
    '',
    '## Сравнение по длине истории без искусственных пропусков',
    '',
    '| Семестров | AP логистическая | AP лес | Brier логистическая | Brier лес |',
    '|---:|---:|---:|---:|---:|',
]
for k in [1,2,3,4,5]:
    z = a[(a.missing_rate == 0) & (a.window_semesters == k)]
    log = z[z.model == 'logistic_regression'].iloc[0]
    forest = z[z.model == 'random_forest'].iloc[0]
    lines.append(f'| {k} | {fmt(log.test_average_precision_mean)} | {fmt(forest.test_average_precision_mean)} | {fmt(log.test_brier_mean)} | {fmt(forest.test_brier_mean)} |')
lines += [
    '',
    '![AP по длине истории](../reports/ablation_ap_by_window.png)',
    '',
    'Для примера на синтетических тестовых данных увеличение окна от 1 до 5 семестров',
    'повысило AP логистической регрессии с 0,754 до 0,816, случайного леса — с 0,731 до 0,819.',
    'Эффект зависит от выбранного генератора данных, не подлежит переносу на реальные наблюдения без отдельной проверки.',
    '',
    '## Сценарии с 25% пропусков',
    '',
    '| Модель | Механизм | AP: 1 семестр | AP: 3 семестра | AP: 5 семестров | Лучшее окно по validation |',
    '|---|---|---:|---:|---:|---:|',
]
for pattern in ['mcar_cells','blocked_terms']:
    for model, title in [('logistic_regression','Логистическая'),('random_forest','Случайный лес')]:
        z = a[(a.missing_rate==.25)&(a.pattern==pattern)&(a.model==model)]
        w = {int(x.window_semesters): x for _,x in z.iterrows()}
        sel = s[(s.missing_rate==.25)&(s.pattern==pattern)&(s.model==model)].iloc[0]
        lines.append(f'| {title} | {pattern} | {fmt(w[1].test_average_precision_mean)} | {fmt(w[3].test_average_precision_mean)} | {fmt(w[5].test_average_precision_mean)} | {int(sel.chosen_window_on_validation)} |')
lines += [
    '',
    '![Устойчивость при потере наблюдений](../reports/ablation_missingness_ap.png)',
    '',
    'При 25% независимо пропущенных измерений случайный лес по validation выбрал окно',
    'в 3 семестра (AP test 0,777); тестовое AP для 5 семестров 0,796.',
    'Выбирать окно 5 только потому, что тестовый показатель выше, **нельзя**:',
    'это означало бы настройку по отложенной выборке.',
    '',
    '## Выбор длины истории по валидации',
    '',
    '| Пропуски | Семейство | Окно (val) | AP тест выбранное | AP тест срез (1) | Δ AP |',
    '|---|---|---:|---:|---:|---:|',
]
for pattern, rate in [('mcar_cells',0.0),('mcar_cells',0.25),('blocked_terms',0.25),('blocked_terms',0.4)]:
    for model, title in [('logistic_regression','Логистическая'),('random_forest','Случайный лес')]:
        z = s[(s.missing_rate == rate) & (s.pattern == pattern) & (s.model == model)].iloc[0]
        label = ('без пропусков' if rate == 0 else f'{pattern}, {rate:.0%}')
        lines.append(f'| {label} | {title} | {int(z.chosen_window_on_validation)} | {fmt(z.test_ap_chosen)} | {fmt(z.test_ap_snapshot)} | {z.test_ap_delta_vs_snapshot:+.3f} |')
lines += [
    '',
    '## Интерпретация',
    '',
    'На смоделированных данных многосеместровые признаки ассоциированы с улучшением AP',
    'и часто меньшим значением Brier score. Повторения с разными масками демонстрируют',
    'чувствительность к типу и уровню пропусков. В некоторых режимах критерий validation',
    'выбирает не максимальную длину окна. Это мотивирует исследование алгоритма',
    'адаптации длины истории под доступный массив наблюдений.',
    '',
    '## Методологические ограничения',
    '',
    '1. Все числа относятся только к одной синтетической генеративной модели.',
    '2. Прогнозируется следующий семестр после завершения текущего.',
    '3. Случайная потеря целого семестра не равна реалистичному механизму MAR/MNAR.',
    '4. Три маски пропусков не образуют независимых когорт, стандартное отклонение по маскам не является доверительным интервалом.',
    '5. Студенты повторяются по временным срезам; не проверялась переносимость на новые группы.',
    '6. Для утверждения научной новизны нужен поиск аналогов, обоснование собственного метода, независимая проверка и сравнение.',
    '',
    '## Следующий шаг',
    '',
    'Сформировать формальную постановку алгоритма **адаптивного окна истории**, который оценивает качество',
    'и полноту предыдущих наблюдений до момента прогноза и выбирает длину окна без доступа к будущим значениям.',
    'Сравнить с оптимизированными фиксированными окнами на независимых по времени и когортам данных.',
    '',
    '## Воспроизводимость',
    '',
    'Конфигурация, точные маски, предсказательные признаки и все 190 оценок доступны в `reports/ablation_*_synthetic.*`.',
    'Протокол эксперимента: `research/07_window_missingness_protocol_v03.md`.',
    'Запуск: `python -m edurisk.ablation --input data/synthetic_long_10_terms.csv --output reports`.',
]
(ROOT / 'research' / '08_window_missingness_results_v03.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

fig,ax=plt.subplots(figsize=(8,4.8))
for family,title in [('logistic_regression','Логистическая регрессия'),('random_forest','Случайный лес')]:
    z = a[(a.missing_rate==0)&(a.model==family)].sort_values('window_semesters')
    ax.plot(z.window_semesters,z.test_average_precision_mean,marker='o',label=title)
ax.set(xlabel='Число семестров в окне',ylabel='Average Precision (test)',title='Влияние длины истории · синтетический эксперимент')
ax.set_xticks([1,2,3,4,5]);ax.set_ylim(.68,.85);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(r/'ablation_ap_by_window.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(figsize=(8.5,4.9))
for k in [1,3,5]:
    z=a[(a.model=='random_forest')&(a.pattern=='mcar_cells')&(a.window_semesters==k)].sort_values('missing_rate')
    ax.plot(z.missing_rate*100,z.test_average_precision_mean,marker='o',label=f'{k} сем.')
ax.set(xlabel='Доля пропущенных отдельных показателей, %',ylabel='Average Precision (test)',title='Устойчивость к пропускам · случайный лес · синтетика')
ax.set_xticks([0,10,25,40]);ax.set_ylim(.6,.85);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(r/'ablation_missingness_ap.png',dpi=170);plt.close(fig)
print('Report and plots generated.')
