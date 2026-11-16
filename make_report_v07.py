"""Generate descriptive charts and Russian summary from v0.7 synthetic experiment."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
R=ROOT/'reports'
s=pd.read_csv(R/'uncertainty_summary_synthetic.csv')
choices=pd.read_csv(R/'uncertainty_selections_synthetic.csv')
mech={'complete':'Без пропусков', 'mar_attendance':'MAR',
      'mnar_value':'MNAR','contiguous_terms':'Пропущенные периоды'}
fam={'logistic_regression':'Логистическая регрессия', 'random_forest':'Случайный лес'}
sel={'selected_nested':'v0.6: pooled-AP', 'selected_fold_mean':'Средняя AP',
     'selected_stability_1':'AP - SE', 'selected_stability_2':'AP - 2 SE',
     'selected_one_se':'Правило 1 SE'}

sub=s[s.policy.eq('selected_stability_1')].sort_values(['mechanism','family'])
labels=[f"{mech[r.mechanism]} · {'LR' if r.family == 'logistic_regression' else 'RF'}" for r in sub.itertuples()]
means=sub.delta_v06_mean.to_numpy()
error=[[max(0,a-b) for a,b in zip(means,sub.delta_v06_ci_lo)],
       [max(0,b-a) for a,b in zip(means,sub.delta_v06_ci_hi)]]
fig,ax=plt.subplots(figsize=(11,6))
ax.barh(labels,means,xerr=error,capsize=3)
ax.axvline(0,linewidth=1)
ax.set_xlabel('Разница AP против селектора v0.6 на ТОЙ ЖЕ выборке')
ax.set_title('EduRisk v0.7: штраф за нестабильность (1 SE), синтетические данные')
ax.invert_yaxis()
fig.tight_layout();fig.savefig(R/'uncertainty_delta_vs_v06_synthetic.png',dpi=160);plt.close(fig)

frame=(choices.melt(id_vars=['seed','mechanism','family'],
 value_vars=['selected_v06','selected_stability_1','selected_stability_2','selected_one_se'],
 var_name='selector',value_name='strategy')
 .groupby(['selector','strategy']).size().unstack(fill_value=0))
fig,ax=plt.subplots(figsize=(10,5))
frame.plot.bar(stacked=True,ax=ax)
ax.set_ylabel('Число выбранных стратегий (из 48 сравнения)')
ax.set_xlabel('Правило выбора')
ax.set_title('Какие стратегии выбирают разные селекторы')
ax.tick_params(axis='x',rotation=15)
fig.tight_layout();fig.savefig(R/'uncertainty_selection_counts_synthetic.png',dpi=160);plt.close(fig)

sub=s[s.policy.isin(['selected_nested','selected_stability_1','selected_stability_2','selected_one_se'])]
lines=['# EduRisk v0.7 — отчёт об устойчивом выборе стратегии','',
'**Область применимости: синтетический исследовательский эксперимент.**','',
'## Условия','',
'- 6 независимых генераций, 90 вымышленных студентов на генерацию; 10 семестров; 4 механизма данных; 2 модели.',
'- Окна истории: 1, 3, 5. Внутренние проверки 6/7/8 семестры; внешняя тестовая оценка — семестр 9.',
'- 48 сравнений, 576 обучений базовых моделей. Один и тот же набор прогнозов используется для всех правил выбора.',
'- Средняя AP и стандартная ошибка вычисляются по 3 внутренним временным валидациям. Стандартная ошибка — эвристическая мера разброса, НЕ доверительный интервал.',
'- Парный процентильный бутстрэп разницы AP: 5000 выборок с возвращением из 6 независимых генераций. Интервалы описательные и только для данного симулятора.',
'',
'## Изменение качества относительно исходного селектора v0.6','',
'Селектор v0.6 пересчитан на тех же данных и окнах, поэтому сравнения парные. Показатели из старого релиза напрямую сюда не переносятся.','',
'| Пропуски | Модель | Правило | Средняя AP | Δ AP против v0.6 | 95% интервал Δ | Улучшений из 6 |',
'|---|---|---|---:|---:|---|---:|']
for r in sub.sort_values(['mechanism','family','policy']).itertuples():
    lines.append(f'| {mech[r.mechanism]} | {fam[r.family]} | {sel[r.policy]} | {r.ap_mean:.4f} | {r.delta_v06_mean:+.4f} | [{r.delta_v06_ci_lo:+.4f}; {r.delta_v06_ci_hi:+.4f}] | {r.positive_vs_v06}/6 |')
lines+=['','## Интерпретация','',
'Заранее выбранные коэффициенты λ=1,2 помогают не во всех сценариях. Использовать внешний тест для пересмотра λ или выбора «наилучшего» метода нельзя без новой независимой проверки. Отрицательные эффекты являются самостоятельным результатом.',
'',
'Внутренняя валидация использует 3 зависимых временных разбиения; штраф SE не является ни строгой гарантией устойчивости, ни статистическим доверительным интервалом. Выбор по одной стандартной ошибке — известная простая эвристика, не заявляемая здесь как научно новый метод.',
'',
'Тестовая метка для следующего периода никак не используется в выборе модели, истории, коэффициентов или метода. Значение метки читается только после итогового прогноза на отложенном тесте.',
'',
'## Научные ограничения и следующий шаг','',
'1. Повторить анализ на независимом реальном обезличенном образовательном наборе, отдельно сформулировав в нём целевой признак и календарную шкалу.',
'2. Увеличить число действительно независимых генераций и внешних временных срезов; проверить зависимость качества от базовой распространённости класса и механизма пропусков.',
'3. Измерить неопределённость **индивидуальных прогнозов** отдельно (разброс вероятностей между моделями / калибровка / чувствительность к пропускам): текущая версия измеряет преимущественно нестабильность качества модели между периодами.',
'4. При доступе к реальным студенческим данным согласовать правовое основание обработки, минимизацию и обезличивание; исключить автоматическое применение результатов к судьбе студентов.',
'',
'## Воспроизведение','',
'```powershell',
'$env:PYTHONPATH="src"',
'python -m edurisk.uncertainty_selection --output reports',
'python make_report_v07.py',
'pytest -q',
'```',
'',
'Для воспроизведения **опубликованных** таблиц используйте значения CLI по умолчанию: 6 seed, 90 студентов и окна 1/3/5. Для расширенного эксперимента передайте `--windows 1,2,3,4,5 --n-students 120 --seeds ...`; результаты нужно хранить отдельно.',
'','## Проверяемые источники','',
'- Scikit-learn: TimeSeriesSplit — https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html',
'- Scikit-learn: one standard error rule — https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html',
'- Scikit-learn: probability calibration — https://scikit-learn.org/stable/modules/calibration.html',
'- OULAD (UCI) — https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset',
'']
(R/'uncertainty_research_report_v07.md').write_text('\n'.join(lines),encoding='utf-8')
print('written report and figures')
