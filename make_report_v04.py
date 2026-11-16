"""Regenerate research report/plots from the saved synthetic v0.4 experiment."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
REPORT = ROOT / 'reports'
SOURCE = REPORT / 'adaptive_comparison_synthetic.csv'
if not SOURCE.is_file():
    raise SystemExit('Run python -m edurisk.adaptive before this script')
r = pd.read_csv(SOURCE).sort_values(['pattern','missing_rate','model'])
selection = pd.read_csv(REPORT / 'adaptive_selections_synthetic.csv')
usage = pd.read_csv(REPORT / 'adaptive_window_usage_synthetic.csv')

model_titles = {'logistic_regression':'Логистическая регрессия', 'random_forest':'Случайный лес'}
pattern_titles = {'blocked_terms':'Пропущенные семестры', 'mcar_cells':'Пропущенные отдельные значения'}
# Show paired deltas, not absolute metrics to make failure cases visible.
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for (model, pattern), frame in r.groupby(['model','pattern']):
    ax.plot(frame.missing_rate * 100, frame.adaptive_minus_fixed_ap_mean, marker='o',
            label=f'{model_titles[model]} — {pattern_titles[pattern]}')
ax.axhline(0, linewidth=1, linestyle='--')
ax.set(xlabel='Искусственные пропуски, %', ylabel='Δ Average Precision (адаптивный − фиксированный)',
       title='EduRisk v0.4 · влияние типа пропусков · синтетические данные')
ax.legend(fontsize=8, loc='best'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(REPORT/'adaptive_delta_ap_synthetic.png', dpi=160); plt.close(fig)

# Distribution of windows in a specified scenario, transparent about global parameter selection.
fig, ax = plt.subplots(figsize=(9, 4.7))
selected = usage[(usage.pattern=='blocked_terms') & (usage.missing_rate == .25)]
for model, frame in selected.groupby('model'):
    means = frame.groupby('window').fraction.mean()
    ax.plot(means.index, means.values*100, marker='o', label=model_titles[model])
ax.set(xlabel='Выбранное окно, семестров', ylabel='Доля прогнозов, %',
       title='EduRisk v0.4 · распределение окон при 25% отсутствующих семестров')
ax.set_xticks([1,2,3,4,5]);ax.set_ylim(bottom=0)
ax.legend();ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(REPORT/'adaptive_selected_windows_synthetic.png', dpi=160);plt.close(fig)

rows=[]
for x in r.itertuples(index=False):
    rows.append('| {} | {} | {:.0f}% | {:.4f} | {:.4f} | {:+.4f} |'.format(
        pattern_titles[x.pattern], model_titles[x.model], x.missing_rate*100,
        x.fixed_test_ap_mean, x.adaptive_test_ap_mean, x.adaptive_minus_fixed_ap_mean))
num_pos = int((r.adaptive_minus_fixed_ap_mean>1e-8).sum())
num_neg = int((r.adaptive_minus_fixed_ap_mean < -1e-8).sum())
num_eq=len(r)-num_pos-num_neg
selections=[]
for x in selection.itertuples(index=False):
    selections.append('| {} | {} | {:.0f}% | {} | `{}` |'.format(
        pattern_titles[x.pattern], model_titles[x.model], x.missing_rate*100,
        x.selected_fixed_window, x.selected_adaptive_gate))
report=f'''# EduRisk v0.4: эксперимент с адаптивной длиной истории

**Статус:** завершённый воспроизводимый эксперимент на *синтетических* данных (не на реальных студентах).  
**Научная специальность:** 1.2.1 «Искусственный интеллект и машинное обучение».  
**Тема:** «Модели и алгоритмы мониторинга и прогнозирования академической успеваемости обучающихся для поддержки принятия решений в образовательном процессе».

## Цель

Разработать правило индивидуального выбора длины истории на основе полноты и изменчивости измерений и сравнить его с единым фиксированным окном, выбранным по validation. Формальная постановка: `research/10_adaptive_mathematical_model_v04.md`.

## Дизайн

- **420** синтетических студентов, **10** семестров; **3 780** строк с меткой следующего семестра; обучение на t=5–7 (**1260** наблюдений), validation на t=8 (**420**), test на t=9 (**420**).
- Окна 1–5; доли пропусков 0%, 10%, 25%, 40%; два механизма (случайные значения и целые отсутствующие семестры), три маски при ненулевом уровне.
- Модели: логистическая регрессия и случайный лес; всего **190 обучений** базовых моделей и **12 конфигураций гейта**, исследованных на validation. Контрольные фиксированные прогнозы **численно совпали с v0.3** по AP/Brier.
- Лучший фиксированный вариант и конфигурация гейта отбирались **только на validation**, отдельно для каждой экспериментально известной комбинации механизма/уровня пропусков/семейства модели. На test гиперпараметры не подбирались.
- Метрики: Average Precision (основная), ROC-AUC, Brier. В таблице показана средняя разность AP по парным маскам, не доверительные интервалы.

## Полное сравнение на test

| Тип пропусков | Базовая модель | Доля | AP: фиксированное окно | AP: адаптивный выбор | Δ AP |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Из **{len(r)}** сравниваемых комбинаций: **{num_pos}** имеют положительную разность AP, **{num_neg}** — отрицательную, **{num_eq}** совпали. Следовательно, **устойчивого преимущества адаптивного правила во всех сценариях нет**. Особенно показательна отрицательная разница у логистической регрессии при 25% пропусков отдельных измерений — результат не следует скрывать.

![Парная разница AP](../reports/adaptive_delta_ap_synthetic.png)

## Выбранные параметры (только по validation)

| Тип пропусков | Модель | Доля | Выбранное фиксированное k | Выбранная конфигурация гейта |
|---|---|---:|---:|---|
{chr(10).join(selections)}

Конфигурация `c0.5_v0.3_p0.8` означает веса $\\alpha=0.5$, $\\beta=0.3$, $\\gamma=0.8$ в формуле качества окна. См. файл математической постановки.

![Частоты выбора окон](../reports/adaptive_selected_windows_synthetic.png)

## Научная интерпретация и ограничения

**Установлено на уровне синтетической симуляции:** построена исполнимая схема индивидуального выбора окна; обеспечена воспроизводимость прежних моделей; показано, что эффект от адаптивности зависит от типа и доли пропусков и семейства модели. Это экспериментальная демонстрация, **не доказательство научной новизны, реального прогностического эффекта или причинности**.

**Что нельзя утверждать:** что алгоритм всегда лучше, что распределение ошибок соответствует реальным университетам, что найденные веса оптимальны для новых когорт или что три повторные маски представляют три независимые выборки.

**Потенциальные уязвимости:** подбор конфигурации на одном validation-семестре; преимущество экспериментатора, знающего долю и тип пропусков; модель генерации, задающая синтетические временные зависимости; непроверенная переносимость на новых обучающихся и реальную MAR/MNAR неполноту; отсутствие независимых повторов синтетических когорт.

## Предлагаемая следующая проверка

1. Провести **nested/rolling-origin** оценку с внешними временными блоками, на каждом блоке переобучать гейт и базовые модели только на прошлом.
2. Сравнить с более сильными конкурентами: фиксированное k=5, validation-selected k, простая смесь прогнозов, вариант с индикаторами пропусков, взвешенное усреднение без отдельного выбора k.
3. Повторить эксперимент на множестве **независимых синтетических когорт**, а затем на законно полученных обезличенных реальных данных и отдельно на новых программах/группах.
4. Изучить калибровку (Brier, calibration curves), устойчивость к нерегулярной истории и потерям MAR/MNAR; оценивать разности с учётом группирования по студентам.
5. Выполнить систематический обзор аналогичных адаптивных алгоритмов прежде, чем формулировать положение научной новизны.

## Файлы для воспроизведения

- `src/edurisk/adaptive.py` — алгоритм и экспериментальный цикл.
- `reports/adaptive_all_candidates_synthetic.csv` — валидированные 12 гейтов и пять фиксированных окон по каждой маске.
- `reports/adaptive_selections_synthetic.csv` — глобальные настройки, отобранные на validation.
- `reports/adaptive_test_comparison_synthetic.csv` — результаты test на каждом seed.
- `reports/adaptive_comparison_synthetic.csv` — агрегаты и парные разности.
- `reports/adaptive_window_usage_synthetic.csv` — частоты выборов k.
- `reports/adaptive_summary_synthetic.json` — метаданные, сетка и методологические ограничения.
- `reports/adaptive_delta_ap_synthetic.png`, `reports/adaptive_selected_windows_synthetic.png` — графики.

Команды:

```powershell
$env:PYTHONPATH = 'src'
python -m edurisk.adaptive --input data/synthetic_long_10_terms.csv --output reports
python make_report_v04.py
pytest -q
```
'''
(ROOT/'research'/'11_adaptive_results_v04.md').write_text(report,encoding='utf-8')
print(f'Saved report; positives={num_pos}, negatives={num_neg}, ties={num_eq}')
